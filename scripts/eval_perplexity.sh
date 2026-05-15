#!/usr/bin/env bash
# Evaluate fluency by computing perplexity under a stronger LM (default GPT-2 large).
set -e

model_name_or_path=gpt2-large
# model_name_or_path=meta-llama/Llama-2-13b-hf
dataset_path=${1:-./test_data/air_decoding/sentiment_140.0_len512_alpha2.jsonl}
device_num=0

python eval_perplexity.py \
  --model_name_or_path $model_name_or_path \
  --dataset_path $dataset_path \
  --device_num $device_num
