import json
import argparse
import subprocess
import os
import sys
import copy
import cv2 
import torch
import numpy as np
import torchvision.io as tvio

sys.path.append(".")

from decord import VideoReader
from typing import Optional
from PIL import Image
from typing import Dict
from pathlib import Path
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from torchvision.transforms import InterpolationMode
from llava.conversation import conv_templates
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import extract_vision_info,fetch_image
from qwen_vl_utils.vision_process import VIDEO_MIN_PIXELS,VIDEO_TOTAL_PIXELS,VIDEO_MAX_PIXELS,FRAME_FACTOR,IMAGE_FACTOR,VIDEO_READER_BACKENDS,smart_resize,logger
from typing import Union, List
from torchvision import  transforms
from vllm import LLM, SamplingParams

def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        if return_video_sample_fps:
            return video, sample_fps
        return video
   
def load_video(vis_path, sample_frames, num_frm=16, max_clip=4):
    try:
        vr = VideoReader(vis_path, num_threads=1)
        total_frame_num = len(vr)
        fps=vr.get_avg_fps()
        
        total_sample = min(num_frm * max_clip, total_frame_num)
        frame_step = max(total_frame_num // total_sample, 1)
        frame_idx = np.arange(0, total_frame_num, frame_step)[:total_sample]
        
        img_array = vr.get_batch(frame_idx).asnumpy()
        img_array = img_array[..., ::-1]  # BGR→RGB
        
        if len(img_array) > sample_frames:
            indices = np.linspace(0, len(img_array)-1, sample_frames, dtype=int)
            img_array = img_array[indices]
        resized_frames = []
        for frame in img_array:
            resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized_frame)
        img_array = np.array(resized_frames)
        print(f'{len(img_array)}', flush=True)
        return img_array,fps
    except Exception as e:
        print(f"Error: {str(e)}", flush=True)
        return np.empty((0,)),30

def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs


class ModelWrapper:
    def __init__(self, model_type: str, model_path: str, sample_frames: int):
        self.model_type = model_type.lower()
        self.sample_frames = sample_frames
        # self.llm = LLM(model=model_path, tokenizer=None, max_model_len=2048,trust_remote_code=True)
        if self.model_type == "videollama":
            #from videollama import VideoLlama  
            return
            self.model = VideoLlama(model_path)
            self.processor = VideoLlama.get_processor()
            
        elif self.model_type =="qwen2.5-vl-7b-instruct":
            # return
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0") 
                      
            self.processor = AutoProcessor.from_pretrained(model_path,use_fast=True)

        elif self.model_type =="qwen2.5-vl-72b-instruct":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto") 
                      
            self.processor = AutoProcessor.from_pretrained(model_path,use_fast=True)
        elif self.model_type =="llava-onevision-qwen2-7b-ov":
            llava_model_args = {
                "multimodal": True,
            }
            self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(model_path, None, self.model_type, device_map= "cuda", attn_implementation=None, **llava_model_args)

            self.model.eval()
            
            
        else:
            raise NotImplementedError(f"Unsupported model: {model_type}")

    def predict(self, video_path: str, question: str, options: Dict[str, str],sample_frames:int) -> str:
        image_tensors = []
        frames,fps= self._extract_frames(video_path)

        
        if self.model_type == "videollama":
            prompt = self._build_prompt(question, options)
            raw_output = self.model.generate(frames, prompt)
            return self._parse_output(raw_output, options)
       
        elif self.model_type == "videochatgpt":
            inputs = self.processor(
                videos=[frames], 
                text=question,
                options=options,
                return_tensors="pt"
            )
            outputs = self.model(**inputs)
            return outputs['prediction']
        
        elif self.model_type=="qwen2.5-vl-7b-instruct":
            option_prompt='\nOptions: '
            for opt in options.keys():
                option_prompt+=opt+'. '+options[opt]+'\n'
            messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{video_path}",
                        "max_pixels": 360 * 420,
                        "nframes": sample_frames,
                    },
                    {"type": "text", "text": f"Question:  {question}\n{option_prompt}.Answer with the option's letter from the given choices directly and only give the best option."},
                ],
            }
        ]
            
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs,video_kwargs = None,[
            torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames],{'fps': fps}
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs
            )
            inputs = inputs.to("cuda")

            # Inference
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=64)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(f'output_text is {output_text}')
            
            return self._parse_output(output_text[0].strip(), options)
        
        elif self.model_type=="qwen2.5-vl-72b-instruct":
            option_prompt='\nOptions: '
            for opt in options.keys():
                option_prompt+=opt+'. '+options[opt]+'\n'
            messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{video_path}",
                        "max_pixels": 360 * 420,
                        "nframes": sample_frames,
                    },
                    {"type": "text", "text": f"Question:  {question}\n{option_prompt}.Answer with the option's letter from the given choices directly and only give the best option."},
                ],
            }
        ]
            
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs,video_kwargs = None,[
            torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames],{'fps': fps}
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs
            )
            inputs = inputs.to("cuda")

            # Inference
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=64)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(f'output_text is {output_text}')
            
            return self._parse_output(output_text[0].strip(), options)
                        
    def _extract_frames(self, video_path: str) :
        frames,fps=load_video(video_path,sample_frames=self.sample_frames)
        return frames,fps

    def _build_prompt(self, question: str, options: Dict[str, str]) -> str:
        options_text = "\n".join([f"{k}. {v}" for k,v in options.items()])
        return f"{question}\n{options_text}\nAnswer:"

    def _parse_output(self, raw: str, options: Dict[str, str]) -> str:
        for c in ['A', 'B', 'C', 'D']:
            if c in raw[:3]:  
                return c
        return "Unknown"
        
def evaluate(args):
    with open(args.annotation) as f:
        samples = json.load(f)
    if os.path.exists(Path(args.output_dir)/"results.json"):
        with open(Path(args.output_dir)/"results.json", "r") as f:
            old_results = json.load(f)
        for sample in samples:
            for res in old_results:
                if res['questionID']==sample['questionID'] and "correct" in res:
                    sample['pred'] = res['pred']
                    sample['correct'] = res['correct']
    model = ModelWrapper(args.model_type, args.model_path, args.sample_frames)
    
    splited_video_dir = Path(args.splited_video_dir)
    splited_video_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import tqdm
    for i, sample in tqdm.tqdm(enumerate(samples)):
        if 'correct' in sample:
            continue
        video_name = f"{Path(sample['video']).stem}_{sample['end_time']}.mp4"
        clip_path = splited_video_dir / video_name
        
        if not clip_path.exists():
            _crop_video(
                src=Path(os.path.join(args.video_root, sample['field'] , sample['video'])),
                dst=clip_path,
                end_time=sample['end_time']
            )
        
        try:
            pred = model.predict(
                video_path=str(clip_path),
                question=sample['question'],
                options=sample['options'],
                sample_frames=args.sample_frames
            )
            sample['pred'] = pred
            sample['correct'] = (pred == sample['answer'])
        except Exception as e:
            print(f"Error processing {sample['questionID']}: {str(e)}")
            sample['pred'] = "ERROR"
        
        _save_results(samples, args.output_dir)
    
    _save_results(samples, args.output_dir)
    print(f"Evaluation completed. Results saved to {args.output_dir}")

def _crop_video(src: Path, dst: Path, end_time: float):
    cmd = [
        "/bin/ffmpeg", "-y",
        "-ss", "0",
        "-t", str(end_time),
        "-i", str(src),
        "-c", "copy",
        str(dst)
    ]
    subprocess.run(cmd, check=True)

def _save_results(data: list, output_dir: str):
    with open(Path(output_dir)/"results.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoQA Evaluation")
    parser.add_argument("--annotation", type=str, required=True,
                      help="Path to constructed annotation JSON")
    parser.add_argument("--video_root", type=str, required=True,
                      help="Root directory of original videos")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for results and clips")
    parser.add_argument("--splited_video_dir", type=str, required=True,
                      help="Path to splited video")
    parser.add_argument("--model_type", type=str, required=True,
                      choices=["llava-onevision-qwen2-7b-ov","qwen2.5-VL-7b-Instruct","qwen2.5-VL-72b-Instruct"],
                      help="Type of model to evaluate")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to model weights/config")
    parser.add_argument("--sample_frames", type=int, required=True,
                      help="the number of frames sampling from the video")
    
    args = parser.parse_args()
    evaluate(args)
