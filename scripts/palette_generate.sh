#!/usr/bin/env bash
# Palette of Language Models (Yang et al., NAACL 2025) with optional LaPA2.
# Supports LLaMA-2 and Pythia/GPT-NeoX backbones.
set -e

# model_name_or_path=meta-llama/Llama-2-7b-hf
# model_name_or_path=EleutherAI/pythia-12b
model_name_or_path=meta-llama/Llama-2-13b-hf

samples=20
task_mode=detoxification   # sentiment | detoxification
length=512

# Palette hyperparameters (paper defaults)
s_main=1.0
s_aux=0.5
s_prime=0.1
t_coef=0.05

alpha=2                    # 0 disables LaPA2
device_num=0
model_short=$(echo $model_name_or_path | sed 's|.*/||')
variant=${model_short}_len${length}_s${s_main}_t${t_coef}_alpha${alpha}

python palette.py \
  --model_name_or_path $model_name_or_path \
  --samples $samples \
  --task_mode $task_mode \
  --length $length \
  --s_main $s_main \
  --s_aux $s_aux \
  --s_prime $s_prime \
  --t_coef $t_coef \
  --alpha $alpha \
  --variant $variant \
  --device_num $device_num
