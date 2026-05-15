#!/usr/bin/env bash
# Evaluate average toxicity using the Perspective API.
# Provide your API key via the PERSPECTIVE_API_KEY environment variable.
set -e

if [ -z "${PERSPECTIVE_API_KEY:-}" ]; then
  echo "Error: set PERSPECTIVE_API_KEY before running eval_toxic.sh" >&2
  exit 1
fi

dataset_path=${1:-./test_data/air_decoding/detoxification_120.0_len50_alpha2.jsonl}

python eval_toxic_batch.py \
  --dataset_path $dataset_path \
  --API_KEY $PERSPECTIVE_API_KEY
