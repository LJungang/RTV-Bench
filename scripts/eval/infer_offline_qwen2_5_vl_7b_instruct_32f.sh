#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="qwen2.5-VL-7b-Instruct"
MODEL_PATH="./ckpts/Qwen2.5-VL-7B-Instruct"
SAMPLE_FRAMES=32

ANNOTATION="./rtv-bench/QA_clips.json"
VIDEO_ROOT="./rtv-bench/video_clips"

python ./scripts/eval/infer_qwen25vl_rtv.py \
  --annotation "${ANNOTATION}" \
  --video_root "${VIDEO_ROOT}" \
  --splited_video_dir "./tmp_clips" \
  --model_type "${MODEL_NAME}" \
  --model_path "${MODEL_PATH}" \
  --sample_frames "${SAMPLE_FRAMES}" \
  --model_name "${MODEL_NAME}" \
  --eval_results_dir "./eval_results" \
  --device "cuda" \
  --torch_dtype "bfloat16" \
  --resume > "./run_log/eval_${MODEL_NAME}_${SAMPLE_FRAMES}.log" 2>"./run_log/eval_${MODEL_NAME}_${SAMPLE_FRAMES}.log.error" &

