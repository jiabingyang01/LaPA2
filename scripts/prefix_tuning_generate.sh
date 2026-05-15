#!/usr/bin/env bash
# Prefix-Tuning baseline (Li & Liang, 2021) with optional LaPA2.
set -e

model_name_or_path=./models/ckpt_for_sentiment_and_topic
samples=50
task_mode=sentiment        # sentiment | topic | detoxification
length=768
alpha=2                    # 0 disables LaPA2
device_num=0
variant=len${length}_alpha${alpha}

python prefix_tuning.py \
  --model_name_or_path $model_name_or_path \
  --samples $samples \
  --task_mode $task_mode \
  --length $length \
  --alpha $alpha \
  --variant $variant \
  --device_num $device_num
