#!/usr/bin/env python3
"""Extract 16 kHz mono WAV files from VoxCeleb2 MP4 clips."""

import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract audio required by a VoxCeleb2 mixture recipe.")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument(
        "--manifest",
        help="Extract only clips referenced by this MuSE manifest.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def manifest_clips(path):
    clips = set()
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) != 10:
                raise ValueError(
                    f"Expected 10 manifest fields, got {len(row)}: {row}")
            clips.add((row[1], row[2], row[3]))
            clips.add((row[5], row[6], row[7]))
    return sorted(clips)


def scan_clips(video_root):
    clips = []
    for split in ("train", "test"):
        split_root = video_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(split_root)
        for path in sorted(split_root.glob("*/*/*.mp4")):
            speaker, video, filename = path.relative_to(split_root).parts
            clips.append((split, speaker, f"{video}/{Path(filename).stem}"))
    if not clips:
        raise RuntimeError(f"No VoxCeleb2 MP4 files found in {video_root}")
    return clips


def extract_clip(job):
    split, speaker, clip, video_root, audio_root, sample_rate = job
    source = video_root / split / speaker / f"{clip}.mp4"
    target = audio_root / split / speaker / f"{clip}.wav"
    if target.is_file():
        return False
    if not source.is_file():
        raise FileNotFoundError(source)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.wav")
    command = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def main():
    args = parse_args()
    video_root = Path(args.video_root)
    audio_root = Path(args.audio_root)
    clips = (manifest_clips(args.manifest)
             if args.manifest else scan_clips(video_root))
    jobs = [(split, speaker, clip, video_root, audio_root, args.sample_rate)
            for split, speaker, clip in clips]

    extracted = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for index, created in enumerate(executor.map(extract_clip, jobs), 1):
            extracted += int(created)
            if index % 1000 == 0 or index == len(jobs):
                print(f"Prepared {index}/{len(jobs)} source audio files")
    print(f"Extracted {extracted} new WAV files under {audio_root}")


if __name__ == "__main__":
    main()
