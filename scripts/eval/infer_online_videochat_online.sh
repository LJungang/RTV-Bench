#!/usr/bin/env bash
set -euo pipefail

##----------Important parameters----------##
MODEL_NAME="VideoChatOnline-4B"
MODEL_PATH="./ckpts/VideoChatOnline-4B"
SAMPLE_FRAMES=64
##----------------------------------------##

ANNOTATION="./rtv-bench/QA_clips.json"
VIDEO_ROOT="./rtv-bench/video_clips"


export PYTHONPATH="./baseline/VideoChat-Online:${PYTHONPATH:-}"



CUDA_VISIBLE_DEVICES=0
python -m scripts.infer.infer_model_rtv \
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
