"""
Palette of Language Models: A Solver for Controlled Text Generation (NAACL 2025)
Reimplementation based on Yang et al., 2025.

Core formula (Eq.8 + Eq.12 + Eq.13 from paper):
  log p(Z=x) ∝ [Σ f(i)*s_i*c_i*log p(A_i=x) + log P_b] / M1
               + t * [Σ f(i)*s'_i*c'_i*log(1 - p(A_i=x))] / M2

  where c_i = 1 + 1/σ(p(A_i=x))
        c'_i = 1 + 1/σ(1 - p(A_i=x))
        σ = sigmoid
        f(i) = 1 for main attribute, -1 for anti attribute
        M1 = 1 + (2 + 1/e) * Σ s_i
        M2 = (2 + 1/e) * Σ s'_i
"""

import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import random
import argparse
import json
import math

# Custom modeling files with LaPA2 support
import modeling_llama
import modeling_gpt_neox

def parse_fraction(value):
    """Parse a fraction string like '1/3' or a float like '0.5'"""
    if '/' in value:
        num, denom = value.split('/')
        return float(num) / float(denom)
    return float(value)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def set_boost(model_type, enabled, alpha, boost_len):
    """Set LaPA2 boost configuration"""
    if model_type == 'llama':
        modeling_llama.set_boost_config(enabled, alpha, boost_len)
    elif model_type == 'gpt_neox':
        modeling_gpt_neox.set_boost_config(enabled, alpha, boost_len)


def palette_generate(args):
    """
    Palette generation for sentiment control and detoxification.
    Uses hard prompts to induce attributes, then combines logits via Palette formula.
    """
    import os
    os.makedirs(os.path.join(args.output_dir, "palette"), exist_ok=True)
    safe_variant = args.variant.replace('/', '_')  # alpha values like 1/2 contain '/'
    f = open(os.path.join(args.output_dir, "palette", f"{args.task_mode}_{safe_variant}.jsonl"), 'w')
    tokenizer = args.tokenizer
    model = args.model

    # Define attribute prompts (from Palette paper Tables 1, 4)
    prompts_by_task = {
        'sentiment': {
            'main_pos': "The following is a positive movie review, with a very positive sentiment and a very positive tone.",
            'main_neg': "The following is a negative movie review, with a very negative sentiment and a very negative tone.",
        },
        'detoxification': {
            'nontoxic': "The following conversation is one that does not perpetuate negative stereotypes, is not threatening and does not contain any sexually explicit or profane language.",
            'positive': "The following conversation is one that is happy and positive.",
            'fluency': "Continue writing after the conversation to make it more fluency and readable.",
            'toxic': "The following text perpetuates negative stereotypes, is threatening or sexually explicit, or contains profane language.",
        }
    }

    for type_id in args.att_type:
        if args.task_mode == 'detoxification' and type_id == '0':
            continue  # skip non-toxic type, we only generate for toxic prompts

        for prompt_text in tqdm(args.prompt):
            with torch.no_grad():
                # ============================================
                # Setup attribute prompts based on task
                # ============================================
                if args.task_mode == 'sentiment':
                    if type_id == '0':  # positive
                        main_prompt = prompts_by_task['sentiment']['main_pos']
                        anti_prompt = prompts_by_task['sentiment']['main_neg']
                    else:  # negative
                        main_prompt = prompts_by_task['sentiment']['main_neg']
                        anti_prompt = prompts_by_task['sentiment']['main_pos']
                    attr_configs = [
                        {'prompt': main_prompt, 'f': 1.0, 's': args.s_main, 's_prime': args.s_prime},
                        {'prompt': anti_prompt, 'f': -1.0, 's': args.s_anti, 's_prime': args.s_prime},
                    ]
                elif args.task_mode == 'detoxification':
                    # Palette uses 3 attributes for detoxification: nontoxic + positive + fluency
                    attr_configs = [
                        {'prompt': prompts_by_task['detoxification']['nontoxic'], 'f': 1.0, 's': args.s_main, 's_prime': args.s_prime},
                        {'prompt': prompts_by_task['detoxification']['positive'], 'f': 1.0, 's': args.s_aux, 's_prime': args.s_prime},
                        {'prompt': prompts_by_task['detoxification']['fluency'], 'f': 1.0, 's': args.s_aux, 's_prime': args.s_prime},
                    ]

                # ============================================
                # Prepare inputs for each attribute branch + base
                # ============================================
                bos = tokenizer.bos_token if tokenizer.bos_token else ""

                # Base branch (no attribute prompt)
                base_input_ids = torch.tensor([tokenizer(bos + prompt_text).input_ids]).long().to(args.device)
                base_input_ids = base_input_ids.expand(args.samples, -1)

                # Attribute branches
                attr_inputs = []
                attr_prefix_lens = []
                for cfg in attr_configs:
                    attr_text = bos + cfg['prompt'] + " " + prompt_text
                    attr_ids = torch.tensor([tokenizer(attr_text).input_ids]).long().to(args.device)
                    attr_ids = attr_ids.expand(args.samples, -1)
                    attr_inputs.append(attr_ids)
                    # Calculate prefix length for LaPA2 boost
                    attr_prefix_lens.append(len(tokenizer.encode(cfg['prompt'])))

                # ============================================
                # Generation loop
                # ============================================
                max_length = args.length
                base_past_kv = None
                attr_past_kvs = [None] * len(attr_configs)
                prev = None

                if args.task_mode != 'detoxification':
                    cur_len = len(tokenizer.encode(prompt_text))
                else:
                    cur_len = 0
                result = base_input_ids[:, base_input_ids.shape[-1] - cur_len:] if cur_len > 0 else torch.zeros(args.samples, 0, dtype=torch.long, device=args.device)

                e_inv = 1.0 / math.e
                t_coef = args.t_coef  # coefficient for complementary event

                while cur_len < max_length:
                    # --- Forward pass for base branch ---
                    if base_past_kv is None:
                        dic_base = model(input_ids=base_input_ids, return_dict=True, use_cache=True)
                        logits_base = dic_base.logits[:, -1, :]
                        base_past_kv = dic_base.past_key_values
                    else:
                        dic_base = model(input_ids=prev, past_key_values=base_past_kv, return_dict=True, use_cache=True)
                        logits_base = dic_base.logits[:, -1, :]
                        base_past_kv = dic_base.past_key_values

                    # --- Forward pass for each attribute branch ---
                    attr_logits_list = []
                    for i, cfg in enumerate(attr_configs):
                        # Enable LaPA2 boost on this attribute's prefix
                        if args.alpha > 0:
                            set_boost(args.model_type, True, args.alpha, attr_prefix_lens[i])
                        if attr_past_kvs[i] is None:
                            dic_attr = model(input_ids=attr_inputs[i], return_dict=True, use_cache=True)
                            attr_logits_list.append(dic_attr.logits[:, -1, :])
                            attr_past_kvs[i] = dic_attr.past_key_values
                        else:
                            dic_attr = model(input_ids=prev, past_key_values=attr_past_kvs[i], return_dict=True, use_cache=True)
                            attr_logits_list.append(dic_attr.logits[:, -1, :])
                            attr_past_kvs[i] = dic_attr.past_key_values
                        # Disable boost after forward
                        if args.alpha > 0:
                            set_boost(args.model_type, False, 0, 0)

                    # ============================================
                    # Palette combination formula (Eq.8 + Eq.12)
                    # ============================================
                    # Convert logits to probabilities
                    prob_base = torch.softmax(logits_base, dim=-1)
                    log_prob_base = torch.log(prob_base + 1e-10)

                    # Main part: Σ f(i)*s_i*c_i*log p(A_i=x) + log P_b
                    main_sum = log_prob_base.clone()  # start with log P_b

                    # Complementary part: t * Σ f(i)*s'_i*c'_i*log(1 - p(A_i=x))
                    comp_sum = torch.zeros_like(log_prob_base)

                    total_s = 0.0
                    total_s_prime = 0.0

                    for i, cfg in enumerate(attr_configs):
                        prob_attr = torch.softmax(attr_logits_list[i], dim=-1)
                        prob_attr = torch.clamp(prob_attr, 1e-10, 1.0 - 1e-10)

                        # c_i = 1 + 1/σ(p(A_i=x))  (Eq.12)
                        sigma_p = torch.sigmoid(prob_attr)
                        c_i = 1.0 + 1.0 / (sigma_p + 1e-10)

                        # c'_i = 1 + 1/σ(1 - p(A_i=x))
                        sigma_1_minus_p = torch.sigmoid(1.0 - prob_attr)
                        c_prime_i = 1.0 + 1.0 / (sigma_1_minus_p + 1e-10)

                        # Main part contribution
                        log_prob_attr = torch.log(prob_attr + 1e-10)
                        main_sum = main_sum + cfg['f'] * cfg['s'] * c_i * log_prob_attr

                        # Complementary part contribution
                        log_1_minus_p = torch.log(1.0 - prob_attr + 1e-10)
                        comp_sum = comp_sum + cfg['f'] * cfg['s_prime'] * c_prime_i * log_1_minus_p

                        total_s += abs(cfg['s'])
                        total_s_prime += abs(cfg['s_prime'])

                    # Normalization (Eq.12)
                    M1 = 1.0 + (2.0 + e_inv) * total_s
                    M2 = (2.0 + e_inv) * total_s_prime if total_s_prime > 0 else 1.0

                    # Final combined logits
                    final_logits = main_sum / M1 + t_coef * comp_sum / M2

                    # Numerical stability
                    final_logits = torch.nan_to_num(final_logits, nan=0.0, posinf=1e4, neginf=-1e4)

                    # Top-k sampling
                    next_token_probs = torch.softmax(final_logits, dim=-1)
                    top_probs, top_indices = torch.topk(next_token_probs, args.topk, dim=-1)

                    try:
                        tmp_prev = torch.multinomial(top_probs, num_samples=1)
                    except:
                        raise Exception("Sampling failed")
                    prev = top_indices.gather(-1, tmp_prev)
                    result = torch.cat((result, prev), dim=-1)

                    cur_len += 1

            # ============================================
            # Save results
            # ============================================
            clean_res = []
            for i in range(args.samples):
                clean_res.append(tokenizer.decode(result[i]))

            if args.task_mode != 'detoxification':
                for i, text in enumerate(clean_res):
                    data = {}
                    data['text'] = text
                    data[args.task_mode] = args.task_att[args.task_mode][type_id]
                    json.dump(data, f)
                    f.write('\n')
            else:
                data = dict()
                data['prompt'] = prompt_text
                data['text'] = dict()
                for i, text in enumerate(clean_res):
                    data['text'][i] = text
                json.dump(data, f)
                f.write('\n')

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default='meta-llama/Llama-2-7b-hf', type=str)
    parser.add_argument("--prompt_sent", default='./dataset/sentiment-imdb/prompt_sent.jsonl', type=str)
    parser.add_argument("--prompt_topic", default='./dataset/topic-agnews/prompt_topic.jsonl', type=str)
    parser.add_argument("--prompt_detoxification", default='./dataset/detoxification-jigsaw/prompt_detoxification.jsonl', type=str)
    parser.add_argument("--output_dir", default='./test_data', type=str)
    parser.add_argument("--length", default=50, type=int)
    parser.add_argument("--samples", default=20, type=int)
    parser.add_argument("--task_mode", default='detoxification', type=str, choices=['sentiment', 'detoxification'])
    parser.add_argument("--att_type", default=['0', '1'])
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--topk", default=200, type=int)
    # Palette-specific hyperparameters
    parser.add_argument("--s_main", default=1.0, type=float, help="Strength of main attribute")
    parser.add_argument("--s_anti", default=1.0, type=float, help="Strength of anti attribute (sentiment only)")
    parser.add_argument("--s_aux", default=0.5, type=float, help="Strength of auxiliary attributes (detox only)")
    parser.add_argument("--s_prime", default=0.1, type=float, help="Strength for complementary event")
    parser.add_argument("--t_coef", default=0.05, type=float, help="Coefficient t for complementary event term")
    # LaPA2 parameters
    parser.add_argument("--alpha", default=0.0, type=parse_fraction, help="LaPA2 boost strength, 0=disabled, supports fraction like 1/3")
    parser.add_argument("--variant", default="", type=str)
    parser.add_argument("--no_cuda", default=False, action="store_true")
    parser.add_argument("--device_num", default='0', type=str)
    args = parser.parse_args()
    args.device = 'cpu' if args.no_cuda else torch.device("cuda:{}".format(args.device_num))

    set_seed(args)

    # Load model based on model type (custom modeling for LaPA2 support)
    print(f"Loading model: {args.model_name_or_path}")
    model_string_lower = args.model_name_or_path.lower()
    if 'llama' in model_string_lower:
        args.model_type = 'llama'
        args.model = modeling_llama.LlamaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16).to(args.device)
    elif 'pythia' in model_string_lower or 'neox' in model_string_lower:
        args.model_type = 'gpt_neox'
        args.model = modeling_gpt_neox.GPTNeoXForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16).to(args.device)
    else:
        raise ValueError(f"Unsupported model: {args.model_name_or_path}. Use LLaMA or Pythia/GPT-NeoX.")
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.tokenizer.pad_token is None:
        args.tokenizer.pad_token = args.tokenizer.eos_token

    # Task-specific setup
    args.task_att = {
        'sentiment': {'0': 'Positive', '1': 'Negative'},
    }

    if args.task_mode == 'sentiment':
        args.att_type = ['0', '1']
        args.prompt = []
        with open(args.prompt_sent, 'r') as pf:
            for line in pf.readlines():
                args.prompt.append(json.loads(line)['prompt'])
    elif args.task_mode == 'detoxification':
        args.att_type = ['0', '1']
        args.prompt = []
        with open(args.prompt_detoxification, 'r') as pf:
            for line in pf.readlines():
                args.prompt.append(json.loads(line)['prompt'])

    print(f"Task: {args.task_mode}, Length: {args.length}, Samples: {args.samples}")
    print(f"s_main={args.s_main}, s_anti={args.s_anti}, s_aux={args.s_aux}, t={args.t_coef}")
    palette_generate(args)
    print("Done!")
