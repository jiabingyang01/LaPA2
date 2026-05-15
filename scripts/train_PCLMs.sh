#!/usr/bin/env bash
# Train prefix-conditioned language models (PC-LMs) used by Air-Decoding
# and Prefix-Tuning. Produces a single GPT-2 checkpoint with six attribute
# prefixes (positive, negative, world, sports, business, science) and a
# second checkpoint for detoxification.
set -e

model_name_or_path=gpt2-medium
device_num=0
output_dir=./ckpt

python train_PCLMs.py \
  --model_name_or_path $model_name_or_path \
  --output_dir $output_dir \
  --device_num $device_num
