#!/usr/bin/env python3

import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Build a zero-based speaker label map from samples.jsonl")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    speakers = set()
    with open(args.samples, "r", encoding="utf-8") as fin:
        for line in fin:
            speakers.update(json.loads(line)["spk"])

    with open(args.output, "w", encoding="utf-8") as fout:
        json.dump(
            {speaker: label
             for label, speaker in enumerate(sorted(speakers))},
            fout,
            indent=2,
        )


if __name__ == "__main__":
    main()
