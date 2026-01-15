import os
import torch

from scripts.infer.videochat_online_wrapper import (
    VideoChatOnlineWrapper,
    build_pixel_values_and_sec,
)

def main():
    video_path = "/hpc2hdd/home/yuxuanzhao/lijungang/RTV-Bench/rtv-bench/videos/driving_cam/DrivingCam_001.mp4"
    model_type = "VideoChatOnline-4B"
    model_path = "./ckpts/VideoChatOnline-4B"
    sample_frames = 64

    assert os.path.exists(video_path), f"Video not found: {video_path}"
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    print("[VideoChatOnline Caption Test]")
    print(f"Video: {video_path}")
    print("=" * 60)

    # 1) load wrapper
    wrapper = VideoChatOnlineWrapper(
        model_type=model_type,
        model_path=model_path,
        sample_frames=sample_frames,
        device="cuda",
        torch_dtype="bfloat16",
        attn_implementation="eager",
        use_time_prompt=True,
    )

    # 2) preprocess video -> pixel_values, num_patches_list, sec
    pixel_values, num_patches_list, sec = build_pixel_values_and_sec(
        video_path=video_path,
        cfg=wrapper.vcfg,
        transform=wrapper.transform,
    )
    if pixel_values.numel() == 0:
        print("[Error] Empty pixel_values. Video decode/sampling failed.")
        return

    pixel_values = pixel_values.to(device="cuda", dtype=wrapper.torch_dtype, non_blocking=True)

    # 3) build caption prompt
    special_image_tokens = wrapper._build_special_image_tokens(n=len(sec), sec=sec)
    subtitle = wrapper.subtitle_template.format(n=len(sec), secs=", ".join(sec))

    prompt = (
        special_image_tokens
        + "\n"
        + subtitle
        + wrapper.system_message
        + "Please describe what is happening in the video.\n"
        + "Be concise.\n"
    )

    gen_cfg = dict(
        num_beams=1,
        max_new_tokens=128,
        min_new_tokens=1,
        do_sample=False,
    )

    # 4) run chat -> raw text
    with torch.inference_mode():
        out = wrapper.model.chat(
            wrapper.tokenizer,
            pixel_values,
            prompt,
            generation_config=gen_cfg,
            num_patches_list=num_patches_list,
            verbose=True,
        )

    print("\n===== Raw Caption Output =====")
    print(str(out).strip())
    print("==============================")

if __name__ == "__main__":
    main()
