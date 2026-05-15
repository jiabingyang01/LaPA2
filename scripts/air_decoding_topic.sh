#!/usr/bin/env bash
# Air-Decoding + LaPA2 for topic control on GPT-2 PC-LMs.
set -e

model_name_or_path=./models/ckpt_for_sentiment_and_topic
samples=50
task_mode=topic
lambda_cs=60.0
length=512
alpha=2
device_num=0
variant=len${length}_alpha${alpha}

python air_decoding.py \
  --model_name_or_path $model_name_or_path \
  --samples $samples \
  --task_mode $task_mode \
  --lambda_cs $lambda_cs \
  --length $length \
  --alpha $alpha \
  --variant $variant \
  --device_num $device_num
