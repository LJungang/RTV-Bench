#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================================================================
# Created by: Li-Jungang
# Email: ljungang.02@gmail.com
# AI Assistance: Yes (GPT-5.2)
# Is_Check: Yes
# Description: inference script for RTV-Bench using Qwen2.5-VL model
#              - generate caption
#              - answer + reason
# ================================================================

import argparse
import json
import os
import re
import subprocess
import math
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import torch
from decord import VideoReader
from tqdm import tqdm

from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# =========================================================
# New qwen_vl_utils-like constants
# =========================================================
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768

FPS = 2.0
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


# =========================================================
# Utilities
# =========================================================
def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _round_to(x: int, base: int) -> int:
    return int(round(x / base) * base)


def _token_to_pixels(token_num: int, merge_size: int = SPATIAL_MERGE_SIZE) -> int:
    return int(token_num * (merge_size ** 2))


def _choose_target_hw(
    h: int,
    w: int,
    per_frame_token_budget: int,
    merge_size: int = SPATIAL_MERGE_SIZE,
    max_ratio: int = MAX_RATIO,
) -> Tuple[int, int]:
    if h <= 0 or w <= 0:
        return h, w

    ratio = h / w
    if ratio > max_ratio:
        h = int(w * max_ratio)
    elif ratio < 1 / max_ratio:
        w = int(h * max_ratio)

    target_pixels = _token_to_pixels(per_frame_token_budget, merge_size=merge_size)
    cur_pixels = h * w
    if cur_pixels <= 0:
        return h, w

    if cur_pixels > target_pixels:
        scale = math.sqrt(target_pixels / cur_pixels)
        new_h = max(merge_size, int(h * scale))
        new_w = max(merge_size, int(w * scale))
    else:
        new_h, new_w = h, w

    new_h = max(merge_size, _round_to(new_h, merge_size))
    new_w = max(merge_size, _round_to(new_w, merge_size))
    return new_h, new_w


def _uniform_indices(total_frames: int, n: int) -> np.ndarray:
    if n <= 0 or total_frames <= 0:
        return np.array([], dtype=np.int64)
    if n >= total_frames:
        return np.arange(total_frames, dtype=np.int64)
    return np.linspace(0, total_frames - 1, n, dtype=np.int64)


def build_option_prompt(options: Dict[str, str]) -> str:
    keys = sorted(options.keys())
    lines = ["Options:"]
    for k in keys:
        lines.append(f"{k}. {options[k]}")
    return "\n".join(lines)


def extract_pred_letter(text: str, options: Dict[str, str]) -> str:
    if not text:
        return "Unknown"
    valid = set(k.upper() for k in options.keys())

    m = re.search(r"(?i)\banswer\s*[:\-]\s*([A-Z])\b", text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    m = re.search(r"\b([A-Z])\b", text.upper())
    if m and m.group(1) in valid:
        return m.group(1)

    c0 = text.strip()[:1].upper()
    if c0 in valid:
        return c0

    return "Unknown"


def parse_answer_and_reason(text: str, options: Dict[str, str]) -> Tuple[str, str]:
    """
    Expected:
      Answer: B
      Reason: ...
    Robust to small format drift.
    """
    if not text:
        return "Unknown", ""

    ans = ""
    reason = ""

    m = re.search(r"(?i)\banswer\s*[:\-]\s*([A-Z])\b", text)
    if m:
        ans = m.group(1).upper()

    m = re.search(r"(?i)\breason\s*[:\-]\s*(.+)", text, flags=re.DOTALL)
    if m:
        reason = m.group(1).strip()

    if not ans:
        ans = extract_pred_letter(text, options)

    if not reason:
        # try strip answer line away
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 2:
            reason = " ".join(lines[1:]).strip()
        else:
            reason = ""

    return ans, reason


# =========================================================
# Video loading (tensor) - Qwen processor-friendly
# =========================================================
def fetch_video(ele: dict, return_video_sample_fps: bool = False):
    """
    Returns:
      video: torch.FloatTensor (T, 3, H, W) in [0,1]
      sample_fps: float
    """
    path = ele.get("video", None)
    if not isinstance(path, str):
        raise ValueError("ele['video'] must be a str path")

    vr = VideoReader(path, num_threads=1)
    total_frames = len(vr)
    src_fps = float(vr.get_avg_fps()) if total_frames > 0 else float(ele.get("fps", FPS))

    min_frames = int(ele.get("min_frames", FPS_MIN_FRAMES))
    max_frames = int(ele.get("max_frames", FPS_MAX_FRAMES))

    if ele.get("nframes", None) is not None:
        nframes = _clamp(int(ele["nframes"]), min_frames, max_frames)
        duration = total_frames / src_fps if src_fps > 0 else 0.0
        sample_fps = (nframes / duration) if duration > 0 else float(ele.get("fps", FPS))
    else:
        target_fps = float(ele.get("fps", FPS))
        duration = total_frames / src_fps if src_fps > 0 else 0.0
        nframes = int(round(duration * target_fps)) if duration > 0 else min_frames
        nframes = _clamp(nframes, min_frames, max_frames)
        sample_fps = target_fps

    frame_idx = _uniform_indices(total_frames, nframes)
    if frame_idx.size == 0:
        video = torch.zeros((0, 3, 0, 0), dtype=torch.float32)
        return (video, float(sample_fps)) if return_video_sample_fps else video

    frames = vr.get_batch(frame_idx).asnumpy()  # (T, H, W, 3), uint8 RGB
    T, H, W, _ = frames.shape

    vmin = int(ele.get("video_min_token_num", VIDEO_MIN_TOKEN_NUM))
    vmax = int(ele.get("video_max_token_num", VIDEO_MAX_TOKEN_NUM))
    frame_factor = int(ele.get("frame_factor", FRAME_FACTOR))

    per_frame_budget = max(vmin, int(vmax / max(T, 1)))
    per_frame_budget = _clamp(per_frame_budget * frame_factor, vmin, vmax)

    new_h, new_w = _choose_target_hw(H, W, per_frame_budget)

    # IMPORTANT: normalize to [0,1]
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() #/ 255.0  # (T,3,H,W)

    if (new_h, new_w) != (H, W):
        video = F.resize(
            video,
            size=[new_h, new_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        # IMPORTANT: bicubic may overshoot
        # video = video.clamp_(0.0, 1.0)

    if return_video_sample_fps:
        return video, float(sample_fps)
    return video


# =========================================================
# Model wrappers (extensible)
# =========================================================
class BaseModelWrapper:
    def __init__(self, model_type: str, model_path: str, sample_frames: int, device: str = "cuda"):
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.sample_frames = int(sample_frames)
        self.device = device

    def caption(self, video_path: str) -> str:
        raise NotImplementedError

    def answer_with_reason(self, video_path: str, question: str, options: Dict[str, str], caption: str) -> Tuple[str, str]:
        raise NotImplementedError


class Qwen25VLWrapper(BaseModelWrapper):
    def __init__(
        self,
        model_type: str,
        model_path: str,
        sample_frames: int,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_pixels: int = 360 * 420,
    ):
        super().__init__(model_type, model_path, sample_frames, device)
        self.max_pixels = int(max_pixels)

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        td = dtype_map.get(torch_dtype.lower(), torch.bfloat16)

        if "72b" in self.model_type:
            device_map = "auto"
        else:
            device_map = "cuda:0" if device.startswith("cuda") else None

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=td,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.model.eval()

    def _prepare_inputs(self, video_path: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        video_tensor, sample_fps = fetch_video(
            {"video": video_path, "nframes": self.sample_frames},
            return_video_sample_fps=True,
        )
        if video_tensor is None or video_tensor.numel() == 0 or video_tensor.shape[0] == 0:
            raise RuntimeError(f"Empty video tensor for {video_path}")

        inputs = self.processor(
            text=[text],
            images=None,
            videos=[video_tensor],  # batch dimension
            padding=True,
            return_tensors="pt",
            fps=float(sample_fps),
        )

        if self.device.startswith("cuda"):
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        return inputs

    @torch.inference_mode()
    def caption(self, video_path: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{video_path}",
                        "nframes": self.sample_frames,
                        "max_pixels": self.max_pixels,
                    },
                    {
                        "type": "text",
                        "text": (
                            "Generate a concise caption for this video clip in 1-2 sentences. "
                            "Mention key entities, actions, and scene context. Do not hallucinate."
                        ),
                    },
                ],
            }
        ]
        inputs = self._prepare_inputs(video_path, messages)
        generated_ids = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)

        in_len = inputs["input_ids"].shape[1]
        out = generated_ids[:, in_len:]
        cap = self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        cap = re.sub(r"\s+", " ", cap).strip()
        return cap

    @torch.inference_mode()
    def answer_with_reason(self, video_path: str, question: str, options: Dict[str, str], caption: str) -> Tuple[str, str]:
        option_prompt = build_option_prompt(options)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{video_path}",
                        "nframes": self.sample_frames,
                        "max_pixels": self.max_pixels,
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Caption: {caption}\n\n"
                            f"Question: {question}\n"
                            f"{option_prompt}\n\n"
                            "Choose the best option and justify briefly.\n"
                            "Output strictly in this format:\n"
                            "Answer: <OPTION_LETTER>\n"
                            "Reason: <ONE-SHORT-PARAGRAPH>\n"
                        ),
                    },
                ],
            }
        ]
        inputs = self._prepare_inputs(video_path, messages)
        generated_ids = self.model.generate(**inputs, max_new_tokens=180, do_sample=False)

        in_len = inputs["input_ids"].shape[1]
        out = generated_ids[:, in_len:]
        txt = self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        pred, reason = parse_answer_and_reason(txt, options)
        return pred, reason


def build_model_wrapper(model_type: str, model_path: str, sample_frames: int, device: str, torch_dtype: str, max_pixels: int):
    mt = model_type.lower()
    if "qwen2.5" in mt and "vl" in mt:
        return Qwen25VLWrapper(
            model_type=model_type,
            model_path=model_path,
            sample_frames=sample_frames,
            device=device,
            torch_dtype=torch_dtype,
            max_pixels=max_pixels,
        )
    raise NotImplementedError(f"Unsupported model_type: {model_type}")


# =========================================================
# Video cropping (fallback)
# =========================================================
def crop_video_ffmpeg(src: Path, dst: Path, end_time: float):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", "0",
        "-t", str(end_time),
        "-i", str(src),
        "-c", "copy",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


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
        old_map = {x.get("questionID"): x for x in old if ("pred" in x or "caption" in x)}
        for s in samples:
            qid = s.get("questionID")
            if qid in old_map:
                for k in ["caption", "pred", "reason", "correct"]:
                    if k in old_map[qid]:
                        s[k] = old_map[qid][k]

    model = build_model_wrapper(
        model_type=args.model_type,
        model_path=args.model_path,
        sample_frames=args.sample_frames,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_pixels=args.max_pixels,
    )

    splited_video_dir = Path(args.splited_video_dir)
    splited_video_dir.mkdir(parents=True, exist_ok=True)

    # caption cache: avoid recaptioning same clip multiple times
    caption_cache: Dict[str, str] = {}

    for sample in tqdm(samples, desc=f"Eval {args.model_name} nframes={args.sample_frames} cap+reason"):
        if ("correct" in sample) and (sample.get("pred") is not None) and (sample.get("caption") is not None):
            continue

        clip_rel = sample.get("clip_path", None)

        if clip_rel:
            # If your VIDEO_ROOT points to rtv-bench/video_clips, clip_rel may already be relative
            clip_path = Path(args.video_root) / clip_rel if not os.path.isabs(clip_rel) else Path(clip_rel)
            if not clip_path.exists():
                # fallback crop from original video_root/<field>/<video>
                src = Path(args.video_root).parent / sample["field"] / sample["video"]
                clip_path = splited_video_dir / Path(clip_rel).name
                crop_video_ffmpeg(src=src, dst=clip_path, end_time=float(sample["end_time"]))
        else:
            video_name = f"{Path(sample['video']).stem}_{sample['end_time']}.mp4"
            clip_path = splited_video_dir / video_name
            if not clip_path.exists():
                src = Path(args.video_root).parent / sample["field"] / sample["video"]
                crop_video_ffmpeg(src=src, dst=clip_path, end_time=float(sample["end_time"]))

        try:
            clip_key = str(clip_path)
            if clip_key in caption_cache:
                sample["caption"] = caption_cache[clip_key]
            else:
                cap = model.caption(video_path=str(clip_path))
                sample["caption"] = cap
                caption_cache[clip_key] = cap

            pred, reason = model.answer_with_reason(
                video_path=str(clip_path),
                question=sample["question"],
                options=sample["options"],
                caption=sample["caption"],
            )
            sample["pred"] = pred
            sample["reason"] = reason
            sample["correct"] = bool(pred == sample.get("answer", ""))

        except Exception as e:
            sample["pred"] = "ERROR"
            sample["reason"] = ""
            sample["caption"] = sample.get("caption", "")
            sample["correct"] = False
            sample["error"] = str(e)

        save_results_json(samples, args.output_json)

    total = len(samples)
    hit = sum(1 for x in samples if x.get("correct") is True)
    acc = hit / total if total else 0.0
    print(f"[Done] saved to: {args.output_json}")
    print(f"[Acc] {hit}/{total} = {acc:.4f}")


def parse_args():
    p = argparse.ArgumentParser("RTV-Bench inference (Qwen2.5-VL): caption + answer(reason)")

    p.add_argument("--annotation", type=str, required=True, help="Path to QA_clips.json")
    p.add_argument("--video_root", type=str, required=True, help="Root directory that contains video clips")
    p.add_argument("--splited_video_dir", type=str, default="./tmp_clips", help="Store cropped clips if needed")

    p.add_argument("--model_type", type=str, required=True, help='e.g., "qwen2.5-VL-7B-Instruct"')
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--sample_frames", type=int, required=True)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--eval_results_dir", type=str, default="./eval_results")
    p.add_argument("--resume", action="store_true")

    # Pass-through / records (messages-side)
    p.add_argument("--max_pixels", type=int, default=360 * 420)

    args = p.parse_args()
    out_dir = Path(args.eval_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output_json = out_dir / f"{args.model_name}_nframes{args.sample_frames}_cap_reason.json"
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
