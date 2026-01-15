# ================================================================
# Created by: Li-Jungang
# Email: ljungang.02@gmail.com
# AI Assistance: Yes (GPT-5.1)
# Is_Check: Yes
# Description: Clip RTV-Bench videos based on QA.json annotations
# according to start_time and end_time, and
# save the clipped videos and new JSON annotations.
# ================================================================

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split RTV-Bench videos into clips based on QA.json."
    )
    parser.add_argument(
        "--qa_json",
        type=str,
        default="./rtv-bench/QA.json",
        help="Path to the original QA.json file."
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="./rtv-bench/videos",
        help="Root directory of original videos."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./rtv-bench/video_clips",
        help="Directory to save video clips and new json."
    )
    parser.add_argument(
        "--output_json_name",
        type=str,
        default="QA_clips.json",
        help="Name of the new JSON file to save clip annotations."
    )
    parser.add_argument(
        "--use_copy",
        action="store_true",
        help="Use ffmpeg -c copy (no re-encode, faster but may fail on some formats)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers for ffmpeg."
    )
    return parser.parse_args()


def load_qa(qa_path: str) -> List[Dict[str, Any]]:
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"QA.json format error: expect list, got {type(data)}")
    return data


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def run_ffmpeg_clip(
    input_video: str,
    output_video: str,
    start_time: float,
    end_time: float,
    use_copy: bool = False
):
    """
    Clip video using ffmpeg from start_time to end_time.
    """
    duration = max(0.0, end_time - start_time)
    if duration <= 0:
        raise ValueError(f"Invalid duration: start={start_time}, end={end_time}")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-i", input_video,
        "-t", str(duration),
    ]
    if use_copy:
        cmd += ["-c", "copy"]
    else:
        cmd += [
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "-2"
        ]
    cmd.append(output_video)

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {input_video} -> {output_video}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )


def process_entry(
    idx: int,
    item: Dict[str, Any],
    video_root: str,
    output_dir: str,
    use_copy: bool,
    output_json_dir: str,
) -> Optional[Tuple[int, Dict[str, Any]]]:
    video_name = item.get("video")
    field = item.get("field", None)
    start_time = float(item.get("start_time", 0.0))
    end_time = float(item.get("end_time", 0.0))

    if not video_name:
        print(f"[Warning] Entry {idx} has no 'video' field, skip.")
        return None

    # Construct input video path
    if field:
        input_video = os.path.join(video_root, field, video_name)
    else:
        input_video = os.path.join(video_root, video_name)

    if not os.path.exists(input_video):
        print(f"[Warning] Video not found: {input_video}, skip this entry.")
        return None

    stem = Path(video_name).stem
    start_str = f"{start_time:06.2f}".replace(".", "")
    end_str = f"{end_time:06.2f}".replace(".", "")
    clip_filename = f"{stem}_{start_str}-{end_str}_{idx:06d}.mp4"
    clip_path = os.path.join(output_dir, clip_filename)

    try:
        print(f"[{idx}] Cutting clip: {input_video} -> {clip_path}")
        run_ffmpeg_clip(
            input_video=input_video,
            output_video=clip_path,
            start_time=start_time,
            end_time=end_time,
            use_copy=use_copy
        )
    except Exception as e:
        print(f"[Error] Failed to cut clip for entry {idx}: {e}")
        return None

    new_item = dict(item)
    new_item["clip_path"] = os.path.relpath(
        clip_path, start=output_json_dir
    )

    return idx, new_item


def main():
    args = parse_args()

    qa_path = args.qa_json
    video_root = args.video_root
    output_dir = args.output_dir
    output_json_path = os.path.join(output_dir,'..', args.output_json_name)
    output_json_dir = os.path.dirname(output_json_path)

    ensure_dir(output_dir)

    qa_data = load_qa(qa_path)
    print(f"Loaded {len(qa_data)} QA entries from {qa_path}")
    print(f"Using {args.num_workers} workers for ffmpeg.")

    results: List[Tuple[int, Dict[str, Any]]] = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_entry,
                idx,
                item,
                video_root,
                output_dir,
                args.use_copy,
                output_json_dir,
            )
            for idx, item in enumerate(qa_data)
        ]

        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                results.append(res)

    results.sort(key=lambda x: x[0])

    new_annotations: List[Dict[str, Any]] = []
    clip_counter = 0

    for _, new_item in results:
        clip_counter += 1
        clip_id = f"clip_{clip_counter:06d}"
        new_item["clip_id"] = clip_id
        new_annotations.append(new_item)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(new_annotations, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Generated {len(new_annotations)} clips.")
    print(f"New annotation JSON saved to: {output_json_path}")


if __name__ == "__main__":
    main()
