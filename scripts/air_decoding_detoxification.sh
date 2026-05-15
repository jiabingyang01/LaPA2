#!/usr/bin/env bash
# Air-Decoding + LaPA2 for detoxification on GPT-2 PC-LMs.
set -e

model_name_or_path=./models/ckpt_for_detoxification
samples=20
task_mode=detoxification
lambda_cs=120.0
length=50
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
