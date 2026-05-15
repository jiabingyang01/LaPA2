#!/usr/bin/env bash
# Evaluate sentiment attribute accuracy.
set -e

model_name_or_path=./models/best_sentiment_classifier
dataset_path=${1:-./test_data/air_decoding/sentiment_140.0_len512_alpha2.jsonl}
device_num=0

python eval_sent_acc.py \
  --model_name_or_path $model_name_or_path \
  --dataset_path $dataset_path \
  --device_num $device_num
