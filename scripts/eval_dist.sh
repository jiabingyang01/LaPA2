#!/usr/bin/env bash
# Evaluate diversity (Dist-1, Dist-2, Dist-3).
set -e

dataset_path=${1:-./test_data/air_decoding/sentiment_140.0_len512_alpha2.jsonl}
model_name_or_path=roberta-large

python eval_dist.py \
  --dataset_path $dataset_path \
  --model_name_or_path $model_name_or_path
