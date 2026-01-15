#!/usr/bin/env bash
set -euo pipefail

LIST_JSON="./filtered_clips.json"

FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"

python - <<'PY'
import json, os, subprocess, sys

LIST_JSON = "./filtered_clips.json"
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

with open(LIST_JSON, "r", encoding="utf-8") as f:
    items = json.load(f)

def ffmpeg_has_libx264():
    try:
        out = subprocess.check_output([FFMPEG_BIN, "-hide_banner", "-encoders"], stderr=subprocess.STDOUT)
        return b"libx264" in out
    except Exception:
        return False

has_x264 = ffmpeg_has_libx264()
print(f"[INFO] ffmpeg = {FFMPEG_BIN}")
print(f"[INFO] libx264 available: {has_x264}")

fail = 0
ok = 0
skip_missing = 0

for it in items:
    path = it["clip_path"]
    if not os.path.exists(path):
        print(f"[SKIP][MISSING] {path}")
        skip_missing += 1
        continue

    tmp = path + ".tmp.mp4"

    if has_x264:
        cmd = [
            FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
            "-i", path,
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-f", "mp4",
            tmp
        ]
    else:
        cmd = [
            FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
            "-i", path,
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "mpeg4",
            "-q:v", "5",
            "-pix_fmt", "yuv420p",
            "-f", "mp4",
            tmp
        ]

    print("[TRANSCODE]", path)
    try:
        subprocess.check_call(cmd)
        os.replace(tmp, path)
        ok += 1
    except subprocess.CalledProcessError as e:
        fail += 1
        print(f"[FAIL] {path} (ffmpeg exit={e.returncode})")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

print(f"[DONE] ok={ok}, fail={fail}, missing={skip_missing}")
if fail > 0:
    sys.exit(2)
PY
