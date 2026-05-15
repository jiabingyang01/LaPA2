"""Prefix-Tuning baseline (Li & Liang, 2021) + optional LaPA2.

Uses the same GPT-2-based PC-LM checkpoints as Air-Decoding, but performs
direct prefix conditioning -- no base branch, no ADR. LaPA2 augments the
prefix attention logits during decoding via the global boost config in
``modeling_gpt2.set_boost_config``.
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
    modeling_gpt2.set_boost_config(enabled=enabled, alpha=alpha, boost_len=boost_len)


def generate(args):
    out_dir = os.path.join(args.output_dir, 'prefix_tuning')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.task_mode}_{args.variant}.jsonl')
    out_f = open(out_path, 'w')

    tokenizer = args.tokenizer
    model = args.model
    hard_prefix = HARD_PREFIX[args.task_mode]
    hard_prefix_lens = {k: len(tokenizer.encode(v)) for k, v in hard_prefix.items()}
    use_soft_prefix = True  # Prefix-Tuning uses the trained continuous embeddings.

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

                if args.task_mode == 'sentiment':
                    prefix_id = int(type_)
                elif args.task_mode == 'topic':
                    prefix_id = int(type_) + 2
                else:
                    prefix_id = int(type_) + 6

                past_key_values = None
                prev = None
                cur_len = prompt_len if args.task_mode != 'detoxification' else 0
                result = input_text[:, input_text.shape[-1] - cur_len:]

                while cur_len < args.length:
                    if args.alpha > 0:
                        boost(True, args.alpha, args.prefix_len)
                    if past_key_values is None:
                        dic = model(
                            input_ids=input_text, return_dict=True, use_cache=True,
                            use_soft_prefix=use_soft_prefix, prefix_id=prefix_id,
                            base_decoding=False, prompt_len=prompt_len,
                            hard_prefix_len=hard_prefix_lens[type_], task_mode=args.task_mode,
                        )
                    else:
                        dic = model(
                            input_ids=prev, past_key_values=past_key_values,
                            return_dict=True, use_cache=True,
                            use_soft_prefix=use_soft_prefix, prefix_id=prefix_id,
                            base_decoding=False, prompt_len=prompt_len,
                            hard_prefix_len=hard_prefix_lens[type_], task_mode=args.task_mode,
                        )
                    boost(False, 0, 0)
                    logits_att, past_key_values = dic.logits[:, -1, :], dic.past_key_values

                    probs = torch.softmax(logits_att, dim=-1)
                    top_probs, top_indices = torch.topk(probs, args.topk, dim=-1)
                    try:
                        tmp_prev = torch.multinomial(top_probs, num_samples=1)
                    except RuntimeError as exc:
                        raise Exception('Sampling failed') from exc
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
    parser.add_argument('--model_name_or_path', required=True, type=str)
    parser.add_argument('--prompt_sent', default='./dataset/sentiment-imdb/prompt_sent.jsonl')
    parser.add_argument('--prompt_topic', default='./dataset/topic-agnews/prompt_topic.jsonl')
    parser.add_argument('--prompt_detoxification', default='./dataset/detoxification-jigsaw/prompt_detoxification.jsonl')
    parser.add_argument('--output_dir', default='./test_data')
    parser.add_argument('--length', default=256, type=int)
    parser.add_argument('--samples', default=20, type=int)
    parser.add_argument('--task_mode', default='sentiment', choices=['sentiment', 'topic', 'detoxification'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--topk', default=200, type=int)
    parser.add_argument('--alpha', default=0.0, type=parse_fraction, help='LaPA2 boost strength. 0 disables LaPA2.')
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
