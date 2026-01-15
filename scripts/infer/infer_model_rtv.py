# ================================================================
# Created by: Li-Jungang
# Email: ljungang.02@gmail.com
# AI Assistance: Yes (GPT-5.2)
# Is_Check: Yes
# Description: inference script for RTV-Bench using different models
# ================================================================
import argparse
import json
import os
import subprocess
import math
import numpy as np
import torch

from tqdm import tqdm
from torchvision.transforms import functional as F
from scripts.utils.video_utils import crop_video_ffmpeg
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from scripts.infer.base_model_wrapper import BaseModelWrapper
def build_model_wrapper(model_type: str, model_path: str, sample_frames: int, device: str, torch_dtype: str):
    mt = model_type.lower()
    if "qwen2.5" in mt and "vl" in mt:
        from scripts.infer.qwen25vl_wrapper import Qwen25VLWrapper
        return Qwen25VLWrapper(
            model_type=model_type,
            model_path=model_path,
            sample_frames=sample_frames,
            device=device,
            torch_dtype=torch_dtype,
        )
    elif "videochat" in mt and "online" in mt:
        from scripts.infer.videochat_online_wrapper import VideoChatOnlineWrapper
        return VideoChatOnlineWrapper(
            model_type=model_type,
            model_path=model_path,
            sample_frames=sample_frames,
            device=device,
            torch_dtype=torch_dtype,
        )
    elif "dispider" in mt :
        from scripts.infer.dispider_wrapper import DispiderWrapper
        return DispiderWrapper(
            model_type=model_type,
            model_path=model_path,
            sample_frames=sample_frames,
            device=device,
            torch_dtype=torch_dtype,
        )
    raise NotImplementedError(f"Unsupported model_type: {model_type}")

# =========================================================
# IO helpers
# =========================================================
def save_results_json(data: list, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================================================
# Evaluation
# =========================================================
def evaluate(args):
    with open(args.annotation, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # resume
    if args.resume and args.output_json.exists():
        with open(args.output_json, "r", encoding="utf-8") as f:
            old = json.load(f)
        old_map = {x.get("questionID"): x for x in old if "pred" in x}
        for s in samples:
            qid = s.get("questionID")
            if qid in old_map and "correct" in old_map[qid]:
                s["pred"] = old_map[qid]["pred"]
                s["correct"] = old_map[qid]["correct"]

    model = build_model_wrapper(
        model_type=args.model_type,
        model_path=args.model_path,
        sample_frames=args.sample_frames,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    splited_video_dir = Path(args.splited_video_dir)
    splited_video_dir.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(samples, desc=f"Eval {args.model_name} nframes={args.sample_frames}"):
        if "correct" in sample and sample.get("pred") is not None and sample.get("pred") != "ERROR":
            continue

        clip_rel = sample.get("clip_path", None)

        if clip_rel:
            clip_path = Path(args.video_root) / clip_rel if not os.path.isabs(clip_rel) else Path(clip_rel)
            if not clip_path.exists():
                src = Path(args.video_root) / sample["field"] / sample["video"]
                clip_path = splited_video_dir / Path(clip_rel).name
                crop_video_ffmpeg(src=src, dst=clip_path, end_time=float(sample["end_time"]))
        else:
            video_name = f"{Path(sample['video']).stem}_{sample['end_time']}.mp4"
            clip_path = splited_video_dir / video_name
            if not clip_path.exists():
                src = Path(args.video_root) / sample["field"] / sample["video"]
                crop_video_ffmpeg(src=src, dst=clip_path, end_time=float(sample["end_time"]))

        try:
            pred = model.predict(
                video_path=str(clip_path),
                question=sample["question"],
                options=sample["options"],
            )
            sample["pred"] = pred
            sample["correct"] = bool(pred == sample.get("answer", ""))
        except Exception as e:
            sample["pred"] = "ERROR"
            sample["correct"] = False
            sample["error"] = str(e)

        # incremental save
        save_results_json(samples, args.output_json)

    total = len(samples)
    hit = sum(1 for x in samples if x.get("correct") is True)
    acc = hit / total if total else 0.0
    print(f"[Done] saved to: {args.output_json}")
    print(f"[Acc] {hit}/{total} = {acc:.4f}")


def parse_args():
    p = argparse.ArgumentParser("RTV-Bench evaluation (Qwen2.5-VL, class-based)")

    p.add_argument("--annotation", type=str, required=True, help="Path to QA_clips.json")
    p.add_argument("--video_root", type=str, required=True, help="Root directory (contains clips or original videos)")
    p.add_argument("--splited_video_dir", type=str, default="./tmp_clips", help="Store cropped clips if needed")

    p.add_argument("--model_type", type=str, required=True,
                   help='e.g., "qwen2.5-VL-7B-Instruct" or "qwen2.5-VL-72B-Instruct"')
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--sample_frames", type=int, required=True)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    p.add_argument("--model_name", type=str, required=True, help="Name used in output filename")
    p.add_argument("--eval_results_dir", type=str, default="./eval_results")
    p.add_argument("--resume", action="store_true")

    args = p.parse_args()
    out_dir = Path(args.eval_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output_json = out_dir / f"{args.model_name}_nframes{args.sample_frames}.json"
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
