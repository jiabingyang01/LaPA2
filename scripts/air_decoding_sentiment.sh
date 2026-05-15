#!/usr/bin/env bash
# Air-Decoding + LaPA2 for sentiment control on GPT-2 PC-LMs.
# Setting alpha=0 reproduces vanilla Air-Decoding.
set -e

model_name_or_path=./models/ckpt_for_sentiment_and_topic
samples=50
task_mode=sentiment
lambda_cs=140
length=512
alpha=2          # LaPA2 boost strength (try 1, 2, 1/2); 0 disables LaPA2
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
