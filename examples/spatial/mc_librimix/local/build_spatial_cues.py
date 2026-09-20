#!/usr/bin/env python3
"""Export standard per-speaker spatial cues from simulated scenarios."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="all")
    return parser.parse_args()


def read_scenarios(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    args = parse_args()
    scenarios = read_scenarios(args.scenario_manifest)
    if args.split != "all":
        scenarios = [row for row in scenarios if row["split"] == args.split]

    # Group the standard cue resources by dataset split.
    indexes = {}
    for scenario in scenarios:
        split = scenario["split"]
        mixture_id = scenario["mixture_id"]
        cue_dir = Path(args.output_root) / split / "cues"
        npy_dir = cue_dir / "spatial_npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        spatial_index = indexes.setdefault(split, {})

        for index in (1, 2):
            speaker = scenario[f"speaker_{index}"]
            payload = {
                # Spatial frontends consume angles in radians.
                "azimuth":
                np.float32(
                    math.radians(float(
                        scenario[f"source_{index}_azimuth_deg"]))),
                "elevation":
                np.float32(
                    math.radians(
                        float(scenario[f"source_{index}_elevation_deg"]))),
                "distance":
                np.float32(scenario[f"source_{index}_distance"]),
            }
            target_path = npy_dir / f"{mixture_id}__{speaker}.npy"
            np.save(target_path, payload)
            spatial_index[f"{mixture_id}::{speaker}"] = {
                "path": str(target_path),
            }

    # Write one mix-speaker cue index for each split.
    for split, spatial_index in indexes.items():
        output = Path(args.output_root) / split / "cues" / "spatial.json"
        with output.open("w", encoding="utf-8") as stream:
            json.dump(spatial_index, stream, indent=2)
        print(f"Saved {len(spatial_index)} spatial cues to {output}")


if __name__ == "__main__":
    main()
