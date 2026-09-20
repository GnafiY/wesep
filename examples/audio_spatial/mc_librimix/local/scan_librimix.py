#!/usr/bin/env python3
"""Build WeSep samples from generated multichannel Libri2Mix waveforms."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix-dir", required=True)
    parser.add_argument("--s1-dir", required=True)
    parser.add_argument("--s2-dir", required=True)
    parser.add_argument("--outfile", required=True)
    return parser.parse_args()


def speaker_ids(mixture_id):
    parts = mixture_id.split("_")
    if len(parts) != 2:
        raise ValueError(
            f"Expected a two-speaker Libri2Mix ID, got: {mixture_id}")
    return [parts[0].split("-")[0], parts[1].split("-")[0]]


def main():
    args = parse_args()
    mix_dir = Path(args.mix_dir)
    source_dirs = [
        Path(args.s1_dir),
        Path(args.s2_dir),
    ]
    if not mix_dir.is_dir():
        raise FileNotFoundError(mix_dir)
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)

    output = Path(args.outfile)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        for mix_path in sorted(mix_dir.glob("*.wav")):
            source_paths = [
                directory / mix_path.name for directory in source_dirs
            ]
            for source_path in source_paths:
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
            speakers = speaker_ids(mix_path.stem)
            sample = {
                "key": mix_path.stem,
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
            count += 1
    print(f"Saved {count} samples to {output}")


if __name__ == "__main__":
    main()
