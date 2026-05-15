"""Air-Decoding + LaPA2.

Reimplements Air-Decoding (Zhong et al., EMNLP 2023) on top of GPT-2-based
prefix-conditioned LMs, with optional Length-Aware Prefix and Prompt Attention
Augmentation (LaPA2).

LaPA2 is realised by writing the per-branch boost length into the global
boost config in ``modeling_gpt2`` immediately before each forward pass:

    - soft-prefix branch (topic, detoxification):  boost_len = prefix_len
    - hard-prefix branch (sentiment):              boost_len = hard_prefix_len
    - base branch (sentiment / topic):             boost_len = prompt_len
    - base branch (detoxification):                no boost (alpha = 0)

Set ``--alpha 0`` to disable LaPA2 and recover vanilla Air-Decoding.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer

import modeling_gpt2
from modeling_gpt2 import GPT2LMHeadModel


def parse_fraction(value):
    if '/' in value:
        num, denom = value.split('/')
        return float(num) / float(denom)
    return float(value)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


HARD_PREFIX = {
    'sentiment': {'0': 'Very positive:', '1': 'Very negative:'},
    'topic': {
        '0': 'World-related:',
        '1': 'Sports-related:',
        '2': 'Business-related:',
        '3': 'Science-related:',
    },
    'detoxification': {'0': 'Very nontoxic:', '1': 'Very toxic:'},
}


TASK_ATT = {
    'sentiment': {'0': 'Positive', '1': 'Negative'},
    'topic': {'0': 'World', '1': 'Sports', '2': 'Business', '3': 'Science'},
    'detoxification': {'0': 'nontoxic', '1': 'toxic'},
}


def boost(enabled, alpha, boost_len):
    """Convenience wrapper around modeling_gpt2.set_boost_config."""
    modeling_gpt2.set_boost_config(enabled=enabled, alpha=alpha, boost_len=boost_len)


def generate(args):
    out_dir = os.path.join(args.output_dir, 'air_decoding')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.task_mode}_{args.lambda_cs}_{args.variant}.jsonl')
    out_f = open(out_path, 'w')

    tokenizer = args.tokenizer
    model = args.model

    hard_prefix = HARD_PREFIX[args.task_mode]
    hard_prefix_lens = {k: len(tokenizer.encode(v)) for k, v in hard_prefix.items()}
    # Soft prefixes are used for topic and detoxification, hard prefixes for sentiment.
    use_soft_prefix = args.task_mode in ('topic', 'detoxification')

    for type_ in args.att_type:
        if args.task_mode == 'detoxification' and type_ == '1':
            continue
        for prompt in tqdm(args.prompt, desc=f'{args.task_mode}/type={type_}'):
            with torch.no_grad():
                prompt_len = len(tokenizer.encode(prompt))
                input_text = torch.tensor(
                    [tokenizer(tokenizer.eos_token + prompt).input_ids]
                ).long().to(args.device)
                input_text = input_text.expand(args.samples, input_text.shape[-1])

                att_input_text = {}
                for item in args.att_type:
                    if use_soft_prefix:
                        att_input_text[item] = torch.tensor(
                            [tokenizer(tokenizer.eos_token + prompt).input_ids]
                        ).long().to(args.device)
                    else:
                        att_input_text[item] = torch.tensor(
                            [tokenizer(tokenizer.eos_token + hard_prefix[item] + prompt).input_ids]
                        ).long().to(args.device)
                    att_input_text[item] = att_input_text[item].expand(
                        args.samples, att_input_text[item].shape[-1]
                    )

                past_key_values = None
                prev = None
                cur_len = prompt_len if args.task_mode != 'detoxification' else 0
                result = input_text[:, input_text.shape[-1] - cur_len:]

                att_dic = {}
                prob = {}
                sigma_ij = {}
                cond_logits = {}

                while cur_len < args.length:
                    # ---- base branch ----
                    base_boost_len = prompt_len if args.task_mode != 'detoxification' else 0
                    if args.alpha > 0 and base_boost_len > 0:
                        boost(True, args.alpha, base_boost_len)
                    if past_key_values is None:
                        dic_base = model(
                            input_ids=input_text, return_dict=True, use_cache=True,
                            use_soft_prefix=False, base_decoding=True,
                            prompt_len=prompt_len, task_mode=args.task_mode,
                        )
                    else:
                        dic_base = model(
                            input_ids=prev, past_key_values=past_key_values,
                            return_dict=True, use_cache=True,
                            use_soft_prefix=False, base_decoding=True,
                            prompt_len=prompt_len, task_mode=args.task_mode,
                        )
                    boost(False, 0, 0)
                    logits_base, past_key_values = dic_base.logits[:, -1, :], dic_base.past_key_values

                    # ---- attribute branches ----
                    first_step = not att_dic
                    for item in args.att_type:
                        if args.task_mode == 'sentiment':
                            prefix_id = int(item)
                        elif args.task_mode == 'topic':
                            prefix_id = int(item) + 2
                        else:
                            prefix_id = int(item) + 6

                        att_boost_len = args.prefix_len if use_soft_prefix else hard_prefix_lens[item]
                        if args.alpha > 0:
                            boost(True, args.alpha, att_boost_len)

                        if first_step:
                            att_dic.setdefault(item, {})
                            att_dic[item]['dict'] = model(
                                input_ids=att_input_text[item], return_dict=True, use_cache=True,
                                use_soft_prefix=use_soft_prefix, prefix_id=prefix_id,
                                base_decoding=False, prompt_len=prompt_len,
                                hard_prefix_len=hard_prefix_lens[item], task_mode=args.task_mode,
                            )
                        else:
                            att_dic[item]['dict'] = model(
                                input_ids=prev, past_key_values=att_dic[item]['past_kv'],
                                return_dict=True, use_cache=True,
                                use_soft_prefix=use_soft_prefix, prefix_id=prefix_id,
                                base_decoding=False, prompt_len=prompt_len,
                                hard_prefix_len=hard_prefix_lens[item], task_mode=args.task_mode,
                            )
                        boost(False, 0, 0)

                        att_dic[item]['logits'] = att_dic[item]['dict'].logits[:, -1, :]
                        att_dic[item]['past_kv'] = att_dic[item]['dict'].past_key_values

                        if first_step:
                            prob[item] = torch.ones(args.samples, 1).to(args.device)
                        else:
                            prob[item] = torch.gather(att_dic[item]['logits_norm'], dim=-1, index=prev)
                        att_dic[item]['logits_norm'] = -1 / torch.log_softmax(att_dic[item]['logits'], dim=-1)

                    # ---- update running likelihood ratios ----
                    if first_step:
                        for i in args.att_type:
                            for j in args.att_type:
                                if i != j:
                                    sigma_ij[i + j] = prob[i] / prob[j]
                    else:
                        for i in args.att_type:
                            for j in args.att_type:
                                if i != j:
                                    sigma_ij[i + j] *= prob[i] / prob[j]

                    # ---- Air-Decoding ADR reconstruction ----
                    logits_norm_base = torch.softmax(logits_base, dim=-1)
                    for i in args.att_type:
                        cond_logits[i] = None
                        for j in args.att_type:
                            term = att_dic[j]['logits_norm'] if j == i else att_dic[j]['logits_norm'] * sigma_ij[j + i]
                            cond_logits[i] = term if cond_logits[i] is None else cond_logits[i] + term
                        cond_logits[i] = att_dic[i]['logits_norm'] / cond_logits[i]
                        cond_logits[i] = torch.nan_to_num(cond_logits[i], nan=0)

                    next_token_logits = logits_norm_base * (cond_logits[type_] ** args.lambda_cs)
                    top_probs, top_indices = torch.topk(next_token_logits, args.topk, dim=-1)
                    try:
                        tmp_prev = torch.multinomial(top_probs, num_samples=1)
                    except RuntimeError as exc:
                        raise Exception('Sampling failed (lambda_cs may be too high)') from exc
                    prev = top_indices.gather(-1, tmp_prev)
                    result = torch.cat((result, prev), dim=-1)
                    cur_len += 1

            decoded = [tokenizer.decode(result[i]) for i in range(args.samples)]
            if args.task_mode != 'detoxification':
                for text in decoded:
                    out_f.write(json.dumps({'text': text, args.task_mode: TASK_ATT[args.task_mode][type_]}) + '\n')
            else:
                out_f.write(json.dumps({
                    'prompt': prompt,
                    'text': {i: t for i, t in enumerate(decoded)},
                }) + '\n')

    out_f.close()
    print(f'Wrote {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True, type=str,
                        help='Path to the PC-LM checkpoint (e.g. ckpt_for_sentiment_and_topic).')
    parser.add_argument('--prompt_sent', default='./dataset/sentiment-imdb/prompt_sent.jsonl')
    parser.add_argument('--prompt_topic', default='./dataset/topic-agnews/prompt_topic.jsonl')
    parser.add_argument('--prompt_detoxification', default='./dataset/detoxification-jigsaw/prompt_detoxification.jsonl')
    parser.add_argument('--output_dir', default='./test_data')
    parser.add_argument('--length', default=512, type=int)
    parser.add_argument('--samples', default=20, type=int)
    parser.add_argument('--task_mode', default='sentiment', choices=['sentiment', 'topic', 'detoxification'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--topk', default=200, type=int)
    parser.add_argument('--lambda_cs', default=140.0, type=float, help='Air-Decoding control strength.')
    parser.add_argument('--alpha', default=0.0, type=parse_fraction,
                        help='LaPA2 boost strength (e.g. 2, 1, 1/2). 0 disables LaPA2.')
    parser.add_argument('--variant', default='', type=str)
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--device_num', default='0', type=str)
    args = parser.parse_args()

    args.device = 'cpu' if args.no_cuda else torch.device(f'cuda:{args.device_num}')
    set_seed(args)

    args.model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path).to(args.device)
    args.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    args.prefix_len = args.model.config.prefix_len

    prompt_file = {
        'sentiment': args.prompt_sent,
        'topic': args.prompt_topic,
        'detoxification': args.prompt_detoxification,
    }[args.task_mode]
    with open(prompt_file) as f:
        args.prompt = [json.loads(line)['prompt'] for line in f]

    args.att_type = {
        'sentiment': ['0', '1'],
        'topic': ['0', '1', '2', '3'],
        'detoxification': ['0', '1'],
    }[args.task_mode]

    generate(args)


if __name__ == '__main__':
    main()
