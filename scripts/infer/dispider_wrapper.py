import os
import math
import copy
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from PIL import Image
from decord import VideoReader
from transformers import StoppingCriteria, StoppingCriteriaList

from scripts.infer.base_model_wrapper import BaseModelWrapper
from scripts.utils.video_utils import build_option_prompt
from scripts.utils.prompt_utils import extract_pred_letter


from dispider.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_ANS_TOKEN,
    DEFAULT_TODO_TOKEN,
)
from dispider.conversation import conv_templates
from dispider.model.builder import load_pretrained_model
from dispider.utils import disable_torch_init
from dispider.mm_utils import tokenizer_image_token, get_model_name_from_path


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops: List[torch.LongTensor]):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # stop when the suffix matches any stop sequence
        for stop in self.stops:
            if stop.numel() == 0:
                continue
            if input_ids.shape[1] >= stop.shape[0] and torch.all(
                stop == input_ids[0][-stop.shape[0]:]
            ).item():
                return True
        return False


class DispiderWrapper(BaseModelWrapper):
    """
    DisPider wrapper for RTV-Bench inference.
    - Loads tokenizer/model/processors via dispider.model.builder.load_pretrained_model
    - Samples video into multi-clips and feeds to model.generate(...)
    - Returns a single option letter (A/B/C/D)
    """

    def __init__(
        self,
        model_type: str,
        model_path: str,
        sample_frames: int,
        device: str = "cuda",
        torch_dtype: str = "float16",
        conv_mode: str = "llava_v1",
        model_base: Optional[str] = None,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        num_beams: int = 1,
        max_new_tokens: int = 256,
        max_clip: int = 32,
        num_frm: int = 16,
    ):
        super().__init__(model_type, model_path, sample_frames, device)

        self.model_path = os.path.expanduser(model_path)
        self.model_base = model_base
        self.conv_mode = conv_mode

        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

        self.max_clip = max_clip
        self.num_frm = num_frm

        # dtype
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype.lower(), torch.float16)

        disable_torch_init()

        model_name = get_model_name_from_path(self.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.model_path, self.model_base, model_name
        )
        self.tokenizer = tokenizer
        self.model = model.eval()

        # image_processor is a tuple: (image_processor, time_tokenizer)
        self.image_processor, self.time_tokenizer = image_processor
        self.image_processor_large = self.image_processor

        if self.time_tokenizer.pad_token is None:
            # keep consistent with your script
            self.time_tokenizer.pad_token = "<pad>"

        # stopping criteria
        stop_words_ids = [torch.tensor(self.tokenizer("<|im_end|>").input_ids, device="cuda")]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # cache tokens
        self.ans_token_ids = self.time_tokenizer(DEFAULT_ANS_TOKEN, return_tensors="pt").input_ids
        self.todo_token_ids = self.time_tokenizer(DEFAULT_TODO_TOKEN, return_tensors="pt").input_ids

    # ------------------------
    # Video / preprocessing utils (from your script)
    # ------------------------
    @staticmethod
    def _get_seq_frames(total_num_frames: int, desired_num_frames: int) -> List[int]:
        seg_size = float(total_num_frames - 1) / desired_num_frames
        seq = []
        for i in range(desired_num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            seq.append((start + end) // 2)
        return seq

    @staticmethod
    def _get_seq_time(vr: VideoReader, frame_idx: List[int], num_clip: int) -> np.ndarray:
        frm_per_clip = len(frame_idx) // num_clip
        key_frame = [
            [frame_idx[i * frm_per_clip], frame_idx[i * frm_per_clip + frm_per_clip - 1]]
            for i in range(num_clip)
        ]
        time = vr.get_frame_timestamp(key_frame)
        return np.hstack([time[:, 0, 0], time[:, 1, 1]])

    @staticmethod
    def _calculate_diff(scene_sep: List[int], start_frame: int) -> List[int]:
        diff = [scene_sep[0] - start_frame]
        for i in range(len(scene_sep) - 1):
            diff.append(scene_sep[i + 1] - scene_sep[i])
        return diff

    def _load_video(
        self,
        vis_path: str,
        scene_sep: List[int],
        num_frm: int,
        max_clip: int,
        sample_frame: Optional[List[Tuple[int, int]]] = None,
    ):
        block_size = 1
        vr = VideoReader(vis_path)
        total_frame_num = len(vr) if sample_frame is None else (sample_frame[0][1] - sample_frame[0][0])
        fps = vr.get_avg_fps()
        total_time = total_frame_num / fps

        if len(scene_sep) == 0:
            num_clip = total_time / num_frm
            num_clip = int(block_size * np.round(num_clip / block_size)) if num_clip > block_size else int(np.round(num_clip))
            num_clip = max(num_clip, 5)
            num_clip = min(num_clip, max_clip)
            total_num_frm = num_frm * num_clip
            start_frame = 0 if sample_frame is None else sample_frame[0][0]
            frame_idx = self._get_seq_frames(total_frame_num, total_num_frm)
        else:
            ref_clip = total_time / num_frm
            ref_clip = int(block_size * np.round(ref_clip / block_size)) if ref_clip > block_size else int(np.round(ref_clip))
            ref_clip = max(ref_clip, 5)
            num_clip = max(len(scene_sep), ref_clip)
            num_clip = min(num_clip, max_clip)
            total_num_frm = num_frm * num_clip
            start_frame = 0 if sample_frame is None else sample_frame[0][0]
            frame_idx = []

            if len(scene_sep) < num_clip:
                diff = self._calculate_diff(scene_sep, start_frame)
                ratio = np.array(diff) / (total_frame_num / num_clip)
                ratio = np.maximum(np.round(ratio), 1)
                intervals = np.array(diff) / ratio
                num_clip = int(np.sum(ratio))
                total_num_frm = num_frm * num_clip
                new_sep = []
                start_ = start_frame
                for i in range(len(diff)):
                    for k in range(int(ratio[i])):
                        new_sep.append(int(start_ + intervals[i] * (k + 1)))
                    start_ = scene_sep[i]
                scene_sep = new_sep
                assert len(scene_sep) == num_clip
            elif len(scene_sep) > max_clip:
                diff = self._calculate_diff(scene_sep, start_frame)
                min_idx = np.argsort(diff[:-1])[: len(scene_sep) - max_clip]
                for i in np.sort(min_idx)[::-1]:
                    del scene_sep[i]

            start_ = start_frame
            for end_frame in scene_sep:
                idx_list = np.linspace(start_, end_frame, num=num_frm, endpoint=False)
                frame_idx.extend([int(_id) for _id in idx_list])
                start_ = end_frame

        time_idx = self._get_seq_time(vr, frame_idx, num_clip)
        img_array = vr.get_batch(frame_idx).asnumpy()  # (n, H, W, 3)

        a, H, W, _ = img_array.shape
        if H != W:
            img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
            img_array = torch.nn.functional.interpolate(img_array, size=(min(H, W), min(H, W)))
            img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

        img_array = img_array.reshape((1, len(frame_idx), img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

        frames = [Image.fromarray(img_array[0, j]) for j in range(img_array.shape[1])]
        return frames, time_idx, num_clip

    def _preprocess_time(self, time: np.ndarray, num_clip: int):
        time = time.reshape(2, num_clip)
        seq = []
        block_size = 1
        for i in range(num_clip):
            start, end = time[:, i]
            start = int(np.round(start))
            end = int(np.round(end))
            sentence = f"This contains a clip sampled in {start} to {end} seconds" + DEFAULT_IMAGE_TOKEN
            sentence = tokenizer_image_token(sentence, self.time_tokenizer, return_tensors="pt")
            seq.append(sentence)
        return seq

    def _preprocess_question(self, questions: List[str]):
        seq = []
        for q in questions:
            sentence = tokenizer_image_token(q + DEFAULT_TODO_TOKEN, self.time_tokenizer, return_tensors="pt")
            seq.append(sentence)
        return seq

    def _process_data(
        self,
        video_path: str,
        scene_sep: List[int],
        question: str,
        candidates_lines: List[str],
    ):
        system = (
            "Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        )

        # model.config.mm_use_im_start_end may exist
        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + system + question + "\n" + "\n".join(candidates_lines)

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], "The best answer is:")
        prompt = conv.get_prompt()

        frames, time_idx, num_clips = self._load_video(
            video_path,
            scene_sep=scene_sep,
            num_frm=self.num_frm,
            max_clip=self.max_clip,
        )

        video = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        video = video.view(num_clips, self.num_frm, *video.shape[1:])

        video_large = self.image_processor_large.preprocess(frames, return_tensors="pt")["pixel_values"]
        video_large = video_large.view(num_clips, self.num_frm, *video_large.shape[1:])[:, :1].contiguous()

        seqs = self._preprocess_time(time_idx, num_clips)
        seqs = torch.nn.utils.rnn.pad_sequence(
            seqs,
            batch_first=True,
            padding_value=self.time_tokenizer.pad_token_id,
        )
        compress_mask = seqs.ne(self.time_tokenizer.pad_token_id)

        qs_tok = self._preprocess_question([question])
        qs_tok = torch.nn.utils.rnn.pad_sequence(
            qs_tok,
            batch_first=True,
            padding_value=self.time_tokenizer.pad_token_id,
        )
        qs_mask = qs_tok.ne(self.time_tokenizer.pad_token_id)

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )[:-1]

        return input_ids, video, video_large, seqs, compress_mask, qs_tok, qs_mask

    # ------------------------
    # RTV-Bench predict
    # ------------------------
    @torch.inference_mode()
    def predict(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        """
        Returns: one of {"A","B","C","D"}; fallback "Unknown"
        """
        # candidates: ["A. ...", "B. ...", ...] in stable order
        letters = sorted(options.keys())
        candidates_lines = [f"{k}. {options[k]}" for k in letters]

        try:
            input_ids, video, video_large, seqs, compress_mask, qs_tok, qs_mask = self._process_data(
                video_path=video_path,
                scene_sep=[], 
                question=question,
                candidates_lines=candidates_lines,
            )
        except Exception:
            return "Unknown"

        # to device
        device = "cuda" if self.device.startswith("cuda") else self.device
        input_ids = input_ids.unsqueeze(0).to(device=device, non_blocking=True)

        video = video.to(dtype=self.torch_dtype, device=device, non_blocking=True)
        video_large = video_large.to(dtype=self.torch_dtype, device=device, non_blocking=True)

        seqs = seqs.to(device=device, non_blocking=True)
        compress_mask = compress_mask.to(device=device, non_blocking=True)
        qs_tok = qs_tok.to(device=device, non_blocking=True)
        qs_mask = qs_mask.to(device=device, non_blocking=True)

        ans_token = self.ans_token_ids.to(device=device, non_blocking=True)
        todo_token = self.todo_token_ids.to(device=device, non_blocking=True)

        output_ids = self.model.generate(
            input_ids,
            images=video,
            images_large=video_large,
            seqs=seqs,
            compress_mask=compress_mask,
            qs=qs_tok,
            qs_mask=qs_mask,
            ans_token=ans_token,
            todo_token=todo_token,
            do_sample=True if self.temperature and self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=self.stopping_criteria,
            use_cache=True,
        )

        # decode
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # extract letter
        pred = extract_pred_letter(outputs, options)
        return pred
