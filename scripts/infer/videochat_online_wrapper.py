import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple, List

import torch
import numpy as np
from decord import VideoReader, cpu
import decord
from PIL import Image

from transformers import AutoTokenizer

from scripts.infer.base_model_wrapper import BaseModelWrapper
from scripts.utils.video_utils import build_option_prompt
from scripts.utils.prompt_utils import extract_pred_letter

from internvl.train.dataset import build_transform
from internvl.train.constants import IMG_CONTEXT_TOKEN

# decord.bridge.set_bridge("torch")


# -----------------------------
# Video preprocessing (aligned to reference code)
# -----------------------------
@dataclass
class VideoPreprocessConfig:
    num_segments: int = 16        # number of sampled frames
    input_size: int = 448         # model.config.force_image_size or model.config.vision_config.image_size
    fps: Optional[float] = None   # if provided, sample by fps; else uniform sampling


def _uniform_sample_indices(num_frames: int, n: int) -> np.ndarray:
    if num_frames <= 0:
        return np.array([], dtype=np.int64)
    if n <= 1:
        return np.array([num_frames // 2], dtype=np.int64)
    idxs = np.linspace(0, num_frames - 1, n, endpoint=True).astype(np.int64)
    return idxs


def build_pixel_values_and_sec(
    video_path: str,
    cfg: VideoPreprocessConfig,
    transform,
) -> Tuple[torch.Tensor, List[int], List[str]]:
    """
    Output aligned with InternVL evaluate code:
      - pixel_values: [N, 3, H, W], where N = sum(num_patches_list)
      - num_patches_list: length = num_frames, each entry = #patches for that frame (here always 1)
      - sec: per-frame timestamps (string list)
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    if num_frames <= 0:
        return torch.empty(0), [], []

    # sample indices
    if cfg.fps is None:
        inds = _uniform_sample_indices(num_frames, cfg.num_segments)
    else:
        # fps sampling: take frames with step = avg_fps / cfg.fps, then uniform downsample to num_segments
        avg_fps = float(vr.get_avg_fps())
        step = max(1, int(round(avg_fps / float(cfg.fps))))
        inds = np.arange(0, num_frames, step, dtype=np.int64)
        if inds.size > cfg.num_segments:
            inds = _uniform_sample_indices(int(inds.size), cfg.num_segments)
            # inds now indexes within the sampled list, map back
            inds = np.arange(0, num_frames, step, dtype=np.int64)[inds]
        elif inds.size < cfg.num_segments:
            # pad by uniform over full range
            inds = _uniform_sample_indices(num_frames, cfg.num_segments)

    if inds.size == 0:
        return torch.empty(0), [], []

    # decode frames -> PIL
    frames = vr.get_batch(inds).asnumpy()  # [T, H, W, 3], uint8
    images = [Image.fromarray(f.astype("uint8")) for f in frames]
    sec = [f"{ts[1]:.1f}" for ts in vr.get_frame_timestamp(inds.tolist())]

    # reference code: patches = [image] (no dynamic_preprocess here)
    pixel_values_list: List[torch.Tensor] = []
    num_patches_list: List[int] = []
    for img in images:
        patches = [img]
        num_patches_list.append(len(patches))  # always 1
        pixel_values_list.extend([transform(p) for p in patches])

    pixel_values = torch.stack(pixel_values_list, dim=0)  # [N, 3, H, W]
    return pixel_values, num_patches_list, sec


# -----------------------------
# Wrapper
# -----------------------------
class VideoChatOnlineWrapper(BaseModelWrapper):
    """
    Wrapper for VideoChatOnline_IT (InternVL VideoChat-Online).
    predict() returns an option letter A/B/C/D.
    """

    def __init__(
        self,
        model_type: str,
        model_path: str,
        sample_frames: int,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        attn_implementation: str = "eager",
        # memory banks
        short_memory_bank: int = 64,
        mid_memory_bank: int = 64,
        long_memory_bank: int = 64,
        # sampling
        fps: Optional[float] = None,
        # prompts
        use_time_prompt: bool = False,   # 是否加入 "Frame{i} at {sec}s" 这种格式
        seed: int = 0,
        # if you need to load config from another ckpt, set base_config_path; else use model_path
        base_config_path: Optional[str] = None,
    ):
        super().__init__(model_type, model_path, sample_frames, device)

        random.seed(seed)
        torch.manual_seed(seed)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype.lower(), torch.bfloat16)

        self.use_time_prompt = bool(use_time_prompt)

        # ---- tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            add_eos_token=False,
            trust_remote_code=trust_remote_code,
            use_fast=False,  # 与参考代码一致
        )
        self.tokenizer.model_max_length = 8192 * 4

        # img context token id (reference code)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        # ---- model/config ----
        from internvl.model.videochat_online import InternVLChatConfig, VideoChatOnline_IT

        cfg_path = base_config_path if base_config_path is not None else model_path
        config = InternVLChatConfig.from_pretrained(cfg_path, trust_remote_code=trust_remote_code)

        self.model = VideoChatOnline_IT.from_pretrained(
            model_path,
            config=config,
            torch_dtype=self.torch_dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        ).eval()

        if device.startswith("cuda"):
            self.model = self.model.cuda()

        # set banks (reference idea)
        if hasattr(self.model, "long_bank"):
            self.model.long_bank = long_memory_bank
        if hasattr(self.model, "mid_bank"):
            self.model.mid_bank = mid_memory_bank
        if hasattr(self.model, "short_bank"):
            self.model.short_bank = short_memory_bank

        # set img_context_token_id on model (reference code)
        self.model.img_context_token_id = self.img_context_token_id

        # ---- image size + transform ----
        image_size = getattr(self.model.config, "force_image_size", None)
        if image_size is None and hasattr(self.model.config, "vision_config"):
            image_size = getattr(self.model.config.vision_config, "image_size", 448)
        if image_size is None:
            image_size = 448

        self.transform = build_transform(is_train=False, input_size=int(image_size))

        # preprocess config
        self.vcfg = VideoPreprocessConfig(
            num_segments=int(sample_frames),
            input_size=int(image_size),
            fps=fps,
        )

        # ---- prompts (aligned to reference code) ----
        self.system_message = (
            "Carefully watch the video and pay attention to the cause and sequence of events, "
            "the detail and movement of objects, and the action and pose of persons. "
            "Based on your observations, select the best option that accurately addresses the question.\n"
        )
        self.question_suffix = "\nOnly give the best option."
        self.subtitle_template = "The video contains {n} frames sampled at {secs} seconds.\n"

        # generation defaults (match reference)
        self.default_generation_config = dict(
            num_beams=1,
            max_new_tokens=16,
            min_new_tokens=1,
            do_sample=False,
        )

    def _build_special_image_tokens(self, n: int, sec: Optional[List[str]] = None) -> str:
        if self.use_time_prompt and sec is not None and len(sec) == n:
            return "\n".join([f"Frame{i+1} at {round(float(sec[i]), 1)}s: <image>" for i in range(n)])
        return "\n".join([f"Frame{i+1}: <image>" for i in range(n)])

    @torch.inference_mode()
    def predict(
        self,
        video_path: str,
        question: str,
        options: Dict[str, str],
        generation_config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> str:
        # generation config
        gc = dict(self.default_generation_config)
        if generation_config:
            gc.update({k: v for k, v in generation_config.items() if v is not None})

        # preprocess video -> pixel_values, num_patches_list, sec
        pixel_values, num_patches_list, sec = build_pixel_values_and_sec(
            video_path=video_path,
            cfg=self.vcfg,
            transform=self.transform,
        )
        if pixel_values.numel() == 0:
            return "Unknown"

        # move to device/dtype (reference code uses bf16)
        device = "cuda" if self.device.startswith("cuda") else self.device
        pixel_values = pixel_values.to(device=device, dtype=self.torch_dtype, non_blocking=True)

        # build question text (aligned to reference evaluate_chat_model)
        special_image_tokens = self._build_special_image_tokens(n=len(sec), sec=sec)

        subtitle = self.subtitle_template.format(n=len(sec), secs=", ".join(sec))
        option_prompt = build_option_prompt(options)

        full_question = (
            special_image_tokens
            + "\n"
            + (subtitle if not self.use_time_prompt else "")
            + self.system_message
            + f"Question: {question}\n"
            + f"{option_prompt}\n"
            + self.question_suffix
        )

        # Some implementations read system_message from model attribute
        if hasattr(self.model, "system_message"):
            self.model.system_message = self.system_message

        pred_text = self.model.chat(
            self.tokenizer,
            pixel_values,
            full_question,
            generation_config=dict(
                num_beams=int(gc["num_beams"]),
                max_new_tokens=int(gc["max_new_tokens"]),
                min_new_tokens=int(gc["min_new_tokens"]),
                do_sample=bool(gc["do_sample"]),
            ),
            num_patches_list=num_patches_list,  # IMPORTANT: list length = num_frames, each entry = 1
            verbose=bool(verbose),
        )

        return extract_pred_letter(str(pred_text).strip(), options)
