import torch

from scripts.infer.base_model_wrapper import BaseModelWrapper
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Dict
from scripts.utils.video_utils import fetch_video,build_option_prompt
from scripts.utils.prompt_utils import extract_pred_letter

class Qwen25VLWrapper(BaseModelWrapper):
    def __init__(
        self,
        model_type: str,
        model_path: str,
        sample_frames: int,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ):
        super().__init__(model_type, model_path, sample_frames, device)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        td = dtype_map.get(torch_dtype.lower(), torch.bfloat16)

        # 7B: pin to cuda:0; 72B: auto
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

    @torch.inference_mode()
    def predict(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        option_prompt = build_option_prompt(options)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{video_path}",
                        "nframes": self.sample_frames,
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Question: {question}\n"
                            f"{option_prompt}\n"
                            "Answer with the option's letter from the given choices directly and only give the best option."
                        ),
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Use new fetch_video
        video_tensor, sample_fps = fetch_video(
            {"video": video_path, "nframes": self.sample_frames},
            return_video_sample_fps=True,
        )
        if video_tensor is None or video_tensor.numel() == 0 or video_tensor.shape[0] == 0:
            return "Unknown"

        inputs = self.processor(
            text=[text],
            images=None,
            videos=[video_tensor],  # batch dimension
            padding=True,
            return_tensors="pt",
            fps=float(sample_fps),
        )

        # move inputs to cuda if needed
        if self.device.startswith("cuda"):
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)

        in_len = inputs["input_ids"].shape[1]
        out = generated_ids[:, in_len:]
        output_text = self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        raw = output_text[0].strip()

        return extract_pred_letter(raw, options)