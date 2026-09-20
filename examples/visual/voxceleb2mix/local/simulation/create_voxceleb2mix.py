#!/usr/bin/env python3
"""Generate min-length VoxCeleb2Mix waveforms and raw visual indexes."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate WeSep-ready audio from a MuSE manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def read_manifest(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if len(row) != 10:
                raise ValueError(
                    f"Expected 10 fields at line {line_number}, got {len(row)}"
                )
            if row[2] == row[6]:
                raise ValueError(
                    f"Mixture speakers must differ at line {line_number}")
            if abs(float(row[4])) > 1e-8:
                raise ValueError(
                    f"The target gain must be 0 dB at line {line_number}")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"Empty mixture manifest: {path}")
    return rows


def read_source(path, sample_rate):
    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz, got {rate} Hz: {path}")
    if waveform.shape[1] != 1:
        raise ValueError(f"Expected mono source audio: {path}")
    return waveform[:, 0]


def mixture_key(row, index):
    digest = hashlib.sha1(",".join(row).encode()).hexdigest()[:12]
    return f"mix_{index:08d}_{digest}"


def generate_mixture(job):
    index, row, audio_root, output_root, sample_rate = job
    split = row[0]
    key = mixture_key(row, index)
    filename = f"{key}.wav"
    paths = [
        output_root / split / name / filename for name in ("mix", "s1", "s2")
    ]
    if all(path.is_file() for path in paths):
        return key

    first_path = audio_root / row[1] / row[2] / f"{row[3]}.wav"
    second_path = audio_root / row[5] / row[6] / f"{row[7]}.wav"
    first = read_source(first_path, sample_rate)
    second = read_source(second_path, sample_rate)
    expected = round(float(row[9]) * sample_rate)
    length = min(len(first), len(second), expected)
    if length <= 0:
        raise ValueError(f"Empty mixture duration for {key}")
    first = first[:length]
    second = second[:length]

    # Match the interference power to the target before applying its dB gain.
    target_power = np.mean(first.astype(np.float64)**2)
    interference_power = np.mean(second.astype(np.float64)**2)
    if target_power <= 0 or interference_power <= 0:
        raise ValueError(f"Silent source audio in mixture {key}")
    scale = (10.0**(float(row[8]) / 20.0) *
             np.sqrt(target_power / interference_power))
    second = second * scale
    mixture = first + second

    # Apply one shared peak normalization so mix remains exactly s1 + s2.
    peak = float(np.max(np.abs(mixture)))
    if peak <= 0:
        raise ValueError(f"Silent mixture: {key}")
    first = (first / peak).astype(np.float32)
    second = (second / peak).astype(np.float32)
    mixture = (first + second).astype(np.float32)

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(paths[0], mixture, sample_rate, subtype="FLOAT")
    sf.write(paths[1], first, sample_rate, subtype="FLOAT")
    sf.write(paths[2], second, sample_rate, subtype="FLOAT")
    return key


def main():
    args = parse_args()
    rows = read_manifest(args.manifest)
    audio_root = Path(args.audio_root)
    video_root = Path(args.video_root)
    output_root = Path(args.output_root)
    jobs = [(index, row, audio_root, output_root, args.sample_rate)
            for index, row in enumerate(rows)]

    # Generate fixed audio while preserving manifest order in the cue indexes.
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        keys = list(executor.map(generate_mixture, jobs))

    indexes = {split: {} for split in ("train", "val", "test")}
    mixtures = {split: [] for split in ("train", "val", "test")}
    seen_keys = set()
    for index, (row, key) in enumerate(zip(rows, keys), 1):
        split = row[0]
        if split not in indexes:
            raise ValueError(f"Unsupported mixture split: {split}")
        if key in seen_keys:
            raise ValueError(f"Duplicate mixture key: {key}")
        seen_keys.add(key)

        # Keep speaker and source identities explicit for downstream scanning.
        mixtures[split].append({
            "key":
            key,
            "speakers": [row[2], row[6]],
            "utterances": [
                f"{row[2]}/{row[3]}",
                f"{row[6]}/{row[7]}",
            ],
        })
        for offset in (1, 5):
            source_split = row[offset]
            speaker = row[offset + 1]
            clip = row[offset + 2]
            video = video_root / source_split / speaker / f"{clip}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
            indexes[split][f"{key}::{speaker}"] = [{
                "utt_id": f"{speaker}/{clip}",
                "path": str(video),
            }]
        if index % 1000 == 0 or index == len(rows):
            print(f"Generated {index}/{len(rows)} mixtures")

    for split, visual_index in indexes.items():
        split_dir = output_root / split
        cue_dir = output_root / split / "cues"
        cue_dir.mkdir(parents=True, exist_ok=True)
        with open(cue_dir / "visual.json", "w", encoding="utf-8") as stream:
            json.dump(visual_index, stream, indent=2)
        with open(split_dir / "mixtures.jsonl", "w",
                  encoding="utf-8") as stream:
            for mixture in mixtures[split]:
                stream.write(json.dumps(mixture) + "\n")

    # Describe the active cue representation consumed by Stage 1.
    with open(output_root / "visual_cue.json", "w",
              encoding="utf-8") as stream:
        json.dump(
            {
                "type": "mp4",
                "format": "raw_video",
                "index": "visual.json",
            },
            stream,
            indent=2)


if __name__ == "__main__":
    main()
