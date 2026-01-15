# ================================================================
# Created by: Li-Jungang
# Email: ljungang.02@gmail.com
# Description: utilities for video processing
# ================================================================
import argparse
import json
import os
import subprocess
import math
import numpy as np
import torch

from pathlib import Path
from typing import Dict, Tuple
from decord import VideoReader
from torchvision.transforms import InterpolationMode
from typing import Dict, Tuple, List, Optional, Any
from torchvision.transforms import functional as F
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768

FPS = 2.0
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
MAX_NUM_WORKERS_FETCH_VIDEO = 8
# =========================================================
# Utilities
# =========================================================
def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def _round_to(x: int, base: int) -> int:
    return int(round(x / base) * base)

def build_option_prompt(options: Dict[str, str]) -> str:
    keys = sorted(options.keys())
    lines = ["Options:"]
    for k in keys:
        lines.append(f"{k}. {options[k]}")
    return "\n".join(lines)

def _uniform_indices(total_frames: int, n: int) -> np.ndarray:
    if n <= 0 or total_frames <= 0:
        return np.array([], dtype=np.int64)
    if n >= total_frames:
        return np.arange(total_frames, dtype=np.int64)
    return np.linspace(0, total_frames - 1, n, dtype=np.int64)

def _token_to_pixels(token_num: int, merge_size: int = SPATIAL_MERGE_SIZE) -> int:
    """
    Practical mapping: token budget ~ number of spatial tokens.
    Approx: tokens ~ (H/merge)*(W/merge) => H*W ~ tokens*(merge^2)
    """
    return int(token_num * (merge_size ** 2))

def _choose_target_hw(
    h: int,
    w: int,
    per_frame_token_budget: int,
    merge_size: int = SPATIAL_MERGE_SIZE,
    max_ratio: int = MAX_RATIO,
) -> Tuple[int, int]:
    """
    Decide resize H,W based on per-frame token budget, keep aspect ratio,
    clamp extreme ratios, align to merge_size.
    """
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

    # do not upscale; only downscale if too large
    if cur_pixels > target_pixels:
        scale = math.sqrt(target_pixels / cur_pixels)
        new_h = max(merge_size, int(h * scale))
        new_w = max(merge_size, int(w * scale))
    else:
        new_h, new_w = h, w

    new_h = max(merge_size, _round_to(new_h, merge_size))
    new_w = max(merge_size, _round_to(new_w, merge_size))
    return new_h, new_w



def fetch_video(ele: dict, return_video_sample_fps: bool = False):
    """
    ele format:
      {
        "video": "/path/to/video.mp4",
        "nframes": 16,           # preferred if provided
        "fps": 2.0,              # used if nframes not provided
        "min_frames": 4,
        "max_frames": 768,
        "video_min_token_num": 128,
        "video_max_token_num": 768,
        "frame_factor": 2,
      }

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
        nframes = int(ele["nframes"])
        nframes = _clamp(nframes, min_frames, max_frames)

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

    frames = vr.get_batch(frame_idx).asnumpy()  # (T, H, W, 3), RGB (decord usually RGB)
    T, H, W, _ = frames.shape

    vmin = int(ele.get("video_min_token_num", VIDEO_MIN_TOKEN_NUM))
    vmax = int(ele.get("video_max_token_num", VIDEO_MAX_TOKEN_NUM))
    frame_factor = int(ele.get("frame_factor", FRAME_FACTOR))

    # per-frame budget heuristic
    per_frame_budget = max(vmin, int(vmax / max(T, 1)))
    per_frame_budget = _clamp(per_frame_budget * frame_factor, vmin, vmax)

    new_h, new_w = _choose_target_hw(H, W, per_frame_budget)

    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() #/ 255.0  # (T,3,H,W)
    if (new_h, new_w) != (H, W):
        video = F.resize(
            video,
            size=[new_h, new_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    if return_video_sample_fps:
        return video, float(sample_fps)
    return video

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