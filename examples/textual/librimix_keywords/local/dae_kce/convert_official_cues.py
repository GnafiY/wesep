#!/usr/bin/env python3
"""Convert official sN/mix DAE cues to WeSep mix::speaker indexes."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--official-cues", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples = {}
    with open(args.samples, encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            samples[sample["key"]] = sample["spk"]

    cues = {}
    with open(args.official_cues, encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            source, mix_key = item["key"].split("/", maxsplit=1)
            source_index = int(source[1:]) - 1
            if mix_key not in samples:
                raise KeyError(f"Official mixture not found: {mix_key}")
            speakers = samples[mix_key]
            if source_index < 0 or source_index >= len(speakers):
                raise IndexError(
                    f"Invalid source slot in cue key: {item['key']}")
            lookup_key = f"{mix_key}::{speakers[source_index]}"
            cues[lookup_key] = {
                key: item[key]
                for key in (
                    "normalized_text",
                    "phn_label",
                    "kw_candidate",
                    "keywords_text",
                ) if key in item
            }

    expected = sum(len(speakers) for speakers in samples.values())
    if len(cues) != expected:
        raise RuntimeError(
            f"Official cue coverage mismatch: {len(cues)} != {expected}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(cues, stream, ensure_ascii=False, separators=(",", ":"))
    print(f"Saved {len(cues)} official textual cues to {output}")


if __name__ == "__main__":
    main()
