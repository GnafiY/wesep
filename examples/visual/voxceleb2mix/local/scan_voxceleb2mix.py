#!/usr/bin/env python3
"""Build WeSep samples from a generated VoxCeleb2Mix dataset view."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--cue-index", required=True)
    parser.add_argument("--outfile", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    cue_index = Path(args.cue_index)
    samples_path = Path(args.outfile)
    mix_dir = dataset_dir / "mix"
    source_dirs = [dataset_dir / "s1", dataset_dir / "s2"]
    if not mix_dir.is_dir():
        raise FileNotFoundError(mix_dir)

    # Load the selected cue index for sample-level validation.
    with open(cue_index, encoding="utf-8-sig") as stream:
        visual_index = json.load(stream)

    # Build samples from explicit metadata instead of parsing file names.
    mixture_index = dataset_dir / "mixtures.jsonl"
    with open(mixture_index, encoding="utf-8-sig") as stream:
        mixtures = [json.loads(line) for line in stream if line.strip()]

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8") as stream:
        for mixture in mixtures:
            key = mixture["key"]
            speakers = mixture["speakers"]
            mix_path = mix_dir / f"{key}.wav"
            source_paths = [
                directory / f"{key}.wav" for directory in source_dirs
            ]
            for source_path in [mix_path, *source_paths]:
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
            for speaker in speakers:
                cue_key = f"{key}::{speaker}"
                if cue_key not in visual_index:
                    raise KeyError(f"Visual cue not found: {cue_key}")

            sample = {
                "key": key,
                "spk": speakers,
                "mix": {
                    "default": [str(mix_path)]
                },
                "src": {
                    speakers[0]: [str(source_paths[0])],
                    speakers[1]: [str(source_paths[1])],
                },
            }
            stream.write(json.dumps(sample) + "\n")
    if not mixtures:
        raise RuntimeError(f"Empty mixture index: {mixture_index}")
    print(f"Saved {len(mixtures)} samples to {samples_path}")


if __name__ == "__main__":
    main()
