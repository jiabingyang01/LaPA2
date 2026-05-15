#!/usr/bin/env bash
# PREADD (Pei et al., 2023) and NegPrompt with optional LaPA2.
# Supports LLaMA-2 and Pythia/GPT-NeoX backbones (the models used in the paper).
set -e

model_name_or_path=meta-llama/Llama-2-7b-hf
# model_name_or_path=meta-llama/Llama-2-13b-hf
# model_name_or_path=EleutherAI/pythia-12b

samples=20
task_mode=detoxification   # sentiment | topic | detoxification
length=512
method=preadd              # preadd | neg_prompting
strength=-1.0              # PREADD lambda (only used when method=preadd)
alpha=2                    # 0 disables LaPA2
boost_base=0               # 1 = also boost the prompt on the base branch
device_num=0
model_short=$(echo $model_name_or_path | sed 's|.*/||')
variant=${model_short}_len${length}_${method}_strength${strength}_alpha${alpha}

extra_args=""
if [ "$boost_base" -eq 1 ]; then
  extra_args="--boost_base"
fi

python preadd.py \
  --model_name_or_path $model_name_or_path \
  --samples $samples \
  --task_mode $task_mode \
  --length $length \
  --method $method \
  --strength $strength \
  --alpha $alpha \
  $extra_args \
  --variant $variant \
  --device_num $device_num
