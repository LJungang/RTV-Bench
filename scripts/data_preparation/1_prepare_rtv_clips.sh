#!/usr/bin/env bash
set -e

python ./scripts/data_preparaation/prepare_rtv_clips.py \
  --qa_json ./rtv-bench/QA.json \
  --video_root ./rtv-bench/videos \
  --output_dir ./rtv-bench/video_clips \
  --output_json_name QA_clips.json \
  --use_copy \
  --num_workers 32