
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import random
import argparse
import json
import math
import os

def parse_fraction(value):
    """Parse a fraction string like '1/3' or a float like '0.5'"""
    if '/' in value:
        num, denom = value.split('/')
        return float(num) / float(denom)
    return float(value)

# Custom modeling files with LaPA2 support
import modeling_llama
import modeling_gpt_neox

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def set_boost(model_type, enabled, alpha, boost_len):
    """Set LaPA2 boost configuration."""
    if model_type == 'llama':
        modeling_llama.set_boost_config(enabled, alpha, boost_len)
    elif model_type == 'gpt_neox':
        modeling_gpt_neox.set_boost_config(enabled, alpha, boost_len)

def generate_preadd(args):
    """
    Methods:
    - preadd: final = base + strength * (prefix - base)
    - neg_prompting: final = prefix (only use prefix branch)

    Both support LaPA2: boost prefix attention weights during generation
    """
    os.makedirs(os.path.join(args.output_dir, "preadd"), exist_ok=True)
    f = open(os.path.join(args.output_dir, "preadd", f"{args.task_mode}_{args.variant}.jsonl"), 'w')
    tokenizer = args.tokenizer
    model = args.model

    # Hard prefixes used as the attribute-conditioning prompt.
    hard_prefix = {}
    for item in args.att_type:
        if args.task_mode == "sentiment":
            hard_prefix[item] = "Very positive:" if item == '0' else "Very negative:"
        elif args.task_mode == "topic":
            hard_prefix[item] = {
                '0': "World-related:",
                '1': "Sports-related:",
                '2': "Business-related:",
                '3': "Science-related:",
            }[item]
        elif args.task_mode == "detoxification":
            if item == '0':
                hard_prefix[item] = "Very nontoxic:"
            else:
                # Long-form toxic prompt from PREADD (Pei et al., 2023).
                hard_prefix[item] = (
                    "The following text perpetuates negative stereotypes, is threatening "
                    "or sexually explicit, or contains profane language.\n\n"
                )

    for type in args.att_type:
        if args.task_mode == "detoxification":
            if type == '0':
                continue
        for prompt in tqdm(args.prompt):
            with torch.no_grad():
                # Prefix branch: hard_prefix + prompt (used by both methods)
                prefix_input = torch.tensor([tokenizer(tokenizer.bos_token + hard_prefix[type] + prompt).input_ids]).long().to(args.device)
                prefix_input = prefix_input.expand(args.samples, prefix_input.shape[-1])

                # Calculate prefix length for LaPA2
                hard_prefix_len = len(tokenizer.encode(hard_prefix[type]))
                # Calculate prompt length for base branch LaPA2 boost
                prompt_len = len(tokenizer.encode(prompt))

                # Base branch: just prompt (only for preadd)
                if args.method == 'preadd':
                    base_input = torch.tensor([tokenizer(tokenizer.bos_token + prompt).input_ids]).long().to(args.device)
                    base_input = base_input.expand(args.samples, base_input.shape[-1])
                    base_past_kv = None

                max_length = args.length
                prefix_past_kv = None
                prev = None

                if args.task_mode != 'detoxification':
                    cur_len = len(tokenizer.encode(prompt))
                else:
                    cur_len = 0
                result = prefix_input[:, prefix_input.shape[-1] - cur_len:]
                attention_records = []
                local_attention_records = {k: [] for k in [2, 4, 6, 8]}

                while cur_len < max_length:
                    if prefix_past_kv is None:
                        # First step: full forward pass
                        # Prefix branch (with LaPA2 boost if alpha > 0)
                        if args.alpha > 0:
                            set_boost(args.model_type, True, args.alpha, hard_prefix_len)
                        dic_prefix = model(input_ids=prefix_input, return_dict=True, use_cache=True, output_attentions=args.save_attention)
                        logits_prefix, prefix_past_kv = dic_prefix.logits[:, -1, :], dic_prefix.past_key_values
                        set_boost(args.model_type, False, 0, 0)

                        if args.save_attention and prompt == args.prompt[0]:
                            att_attentions = torch.stack(dic_prefix.attentions, dim=0)
                            prefix_attentions_sum = torch.sum(att_attentions[:, :, :, -1, :hard_prefix_len], dim=-1)
                            prefix_attentions_sum = prefix_attentions_sum.mean(dim=0).mean(dim=0).mean(dim=0)
                            attention_records.append(prefix_attentions_sum.item())
                            avg_attn = att_attentions[:, :, :, -1, :].mean(dim=(0, 1, 2))
                            non_prefix_attn = avg_attn[hard_prefix_len:]
                            non_prefix_sum = non_prefix_attn.sum().item()
                            for k in [2, 4, 6, 8]:
                                if non_prefix_sum > 1e-10 and len(non_prefix_attn) >= k:
                                    local_attention_records[k].append(non_prefix_attn[-k:].sum().item() / non_prefix_sum)
                                else:
                                    local_attention_records[k].append(non_prefix_attn.sum().item() / max(non_prefix_sum, 1e-10))

                        # Base branch (only for preadd)
                        if args.method == 'preadd':
                            # Optional LaPA2 boost on base branch's prompt
                            if args.boost_base and args.alpha > 0:
                                set_boost(args.model_type, True, args.alpha, prompt_len)
                            dic_base = model(input_ids=base_input, return_dict=True, use_cache=True)
                            logits_base, base_past_kv = dic_base.logits[:, -1, :], dic_base.past_key_values
                            if args.boost_base:
                                set_boost(args.model_type, False, 0, 0)
                    else:
                        # Subsequent steps: use KV-cache
                        # Prefix branch (with LaPA2 boost if alpha > 0)
                        if args.alpha > 0:
                            set_boost(args.model_type, True, args.alpha, hard_prefix_len)
                        dic_prefix = model(input_ids=prev, past_key_values=prefix_past_kv, return_dict=True, use_cache=True, output_attentions=args.save_attention)
                        logits_prefix, prefix_past_kv = dic_prefix.logits[:, -1, :], dic_prefix.past_key_values
                        set_boost(args.model_type, False, 0, 0)

                        if args.save_attention and prompt == args.prompt[0]:
                            att_attentions = torch.stack(dic_prefix.attentions, dim=0)
                            prefix_attentions_sum = torch.sum(att_attentions[:, :, :, -1, :hard_prefix_len], dim=-1)
                            prefix_attentions_sum = prefix_attentions_sum.mean(dim=0).mean(dim=0).mean(dim=0)
                            attention_records.append(prefix_attentions_sum.item())
                            avg_attn = att_attentions[:, :, :, -1, :].mean(dim=(0, 1, 2))
                            non_prefix_attn = avg_attn[hard_prefix_len:]
                            non_prefix_sum = non_prefix_attn.sum().item()
                            for k in [2, 4, 6, 8]:
                                if non_prefix_sum > 1e-10 and len(non_prefix_attn) >= k:
                                    local_attention_records[k].append(non_prefix_attn[-k:].sum().item() / non_prefix_sum)
                                else:
                                    local_attention_records[k].append(non_prefix_attn.sum().item() / max(non_prefix_sum, 1e-10))

                        # Base branch (only for preadd)
                        if args.method == 'preadd':
                            # Optional LaPA2 boost on base branch's prompt
                            if args.boost_base and args.alpha > 0:
                                set_boost(args.model_type, True, args.alpha, prompt_len)
                            dic_base = model(input_ids=prev, past_key_values=base_past_kv, return_dict=True, use_cache=True)
                            logits_base, base_past_kv = dic_base.logits[:, -1, :], dic_base.past_key_values
                            if args.boost_base:
                                set_boost(args.model_type, False, 0, 0)

                    # Compute final logits
                    if args.method == 'neg_prompting':
                        # neg_prompting: only use prefix branch (with LaPA2 if enabled)
                        final_logits = logits_prefix
                    else:
                        # PreAdD: final = base + strength * (prefix - base)
                        diff = logits_prefix - logits_base
                        final_logits = logits_base + args.strength * diff

                    final_logits = torch.nan_to_num(final_logits, nan=0.0, posinf=1e4, neginf=-1e4)

                    # Top-k sampling
                    next_token_logits = torch.softmax(final_logits, dim=-1)
                    top_probs, top_indices = torch.topk(next_token_logits, args.topk, dim=-1)

                    try:
                        tmp_prev = torch.multinomial(top_probs, num_samples=1)
                    except:
                        raise Exception("Sampling failed")
                    prev = top_indices.gather(-1, tmp_prev)
                    result = torch.cat((result, prev), dim=-1)

                    cur_len = cur_len + 1

            if args.save_attention and prompt == args.prompt[0]:
                att_save_data = {
                    "task_mode": args.task_mode,
                    "target_type": type,
                    "method": args.method,
                    "prompt": prompt,
                    "hard_prefix": hard_prefix[type],
                    "alpha": args.alpha,
                    "strength": args.strength,
                    "variant": args.variant,
                    "attention": attention_records,
                    "local_attention": local_attention_records
                }
                _plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot")
                os.makedirs(_plot_dir, exist_ok=True)
                att_save_path = os.path.join(_plot_dir, "attention_{}_{}_{}_LaPA2.json".format(args.method, args.task_mode, type))
                with open(att_save_path, 'w') as att_f:
                    json.dump(att_save_data, att_f, indent=2)
                print(f"Attention saved to {att_save_path}")

            clean_res = []
            for i in range(args.samples):
                clean_res.append(tokenizer.decode(result[i]))

            if args.task_mode != 'detoxification':
                for i, text in enumerate(clean_res):
                    data = {}
                    data['text'] = text
                    data[args.task_mode] = args.task_att[args.task_mode][type]
                    json.dump(data, f)
                    f.write('\n')
            else:
                data = dict()
                data['prompt'] = prompt
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
    parser.add_argument("--task_mode", default='sentiment', type=str, choices=['sentiment', 'topic', 'detoxification'])
    parser.add_argument("--att_type", default=['0', '1'])
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--topk", default=200, type=int)
    parser.add_argument("--method", default='preadd', type=str, choices=['preadd', 'neg_prompting'], help="preadd or neg_prompting")
    parser.add_argument("--strength", default=1.0, type=float, help="PreAdD control strength (lambda)")
    parser.add_argument("--alpha", default=0.0, type=parse_fraction, help="LaPA2 boost strength, 0=disabled, supports fraction like 1/3")
    parser.add_argument("--boost_base", default=False, action="store_true", help="Enable LaPA2 boost on base branch's prompt")
    parser.add_argument("--variant", default="", type=str)
    parser.add_argument("--save_attention", default=False, action="store_true", help="save prefix attention data to file for plotting")
    parser.add_argument("--no_cuda", default=False, action="store_true")
    parser.add_argument("--device_num", default='0', type=str)
    args = parser.parse_args()
    args.device = 'cpu' if args.no_cuda else torch.device("cuda:{}".format(args.device_num))

    set_seed(args)

    # Load model based on model_name_or_path
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

    if args.task_mode == 'sentiment':
        args.att_type = ['0', '1']
        prompt_list = list()
        f = open(args.prompt_sent, 'r')
        for item in f.readlines():
            dic = json.loads(item)
            prompt = dic['prompt']
            prompt_list.append(prompt)
        args.prompt = prompt_list

    elif args.task_mode == 'topic':
        args.att_type = ['0', '1', '2', '3']
        prompt_list = list()
        f = open(args.prompt_topic, 'r')
        for item in f.readlines():
            dic = json.loads(item)
            prompt = dic['prompt']
            prompt_list.append(prompt)
        args.prompt = prompt_list

    elif args.task_mode == 'detoxification':
        args.att_type = ['0', '1']
        prompt_list = list()
        f = open(args.prompt_detoxification, 'r')
        for item in f.readlines():
            dic = json.loads(item)
            prompt = dic['prompt']
            prompt_list.append(prompt)
        args.prompt = prompt_list

    task_att = dict()
    task_att['sentiment'] = dict()
    task_att['sentiment']['0'] = 'Positive'
    task_att['sentiment']['1'] = 'Negative'
    task_att['topic'] = dict()
    task_att['topic']['0'] = 'World'
    task_att['topic']['1'] = 'Sports'
    task_att['topic']['2'] = 'Business'
    task_att['topic']['3'] = 'Science'
    task_att['detoxification'] = dict()
    task_att['detoxification']['0'] = 'nontoxic'
    task_att['detoxification']['1'] = 'toxic'
    args.task_att = task_att

    generate_preadd(args)
