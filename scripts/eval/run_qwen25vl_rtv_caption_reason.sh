#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="qwen2.5-VL-7b-Instruct"
MODEL_PATH="./ckpts/Qwen2.5-VL-7B-Instruct"
SAMPLE_FRAMES=8

ANNOTATION="./rtv-bench/QA_clips.json"

# 你的 clip 实际在这里：rtv-bench/video_clips/xxx.mp4
VIDEO_ROOT="./rtv-bench/video_clips"

# 日志目录
mkdir -p ./run_log
python ./scripts/eval/infer_valid_qwen25vl_rtv_based_caption.py \
  --annotation "${ANNOTATION}" \
  --video_root "${VIDEO_ROOT}" \
  --splited_video_dir "./tmp_clips" \
  --model_type "${MODEL_NAME}" \
  --model_path "${MODEL_PATH}" \
  --sample_frames "${SAMPLE_FRAMES}" \
  --model_name "${MODEL_NAME}" \
  --eval_results_dir "./eval_results/valid" \
  --device "cuda" \
  --torch_dtype "bfloat16" \
  --max_pixels $((360*420)) \
  --resume \
  > "./run_log/eval_${MODEL_NAME}_${SAMPLE_FRAMES}_based_caption.log" \
  2> "./run_log/eval_${MODEL_NAME}_${SAMPLE_FRAMES}_based_caption.log.error" &
