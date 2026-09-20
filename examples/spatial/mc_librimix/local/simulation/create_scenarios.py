#!/usr/bin/env python3
"""Create deterministic room scenarios for existing Libri2Mix mixtures."""

import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import math
from pathlib import Path

import numpy as np
import yaml

FIELDS = [
    "version",
    "split",
    "mixture_id",
    "speaker_1",
    "speaker_2",
    "scenario_seed",
    "room_x",
    "room_y",
    "room_z",
    "rt60",
    "array_x",
    "array_y",
    "array_z",
    "array_rotation_deg",
    "source_1_x",
    "source_1_y",
    "source_1_z",
    "source_1_azimuth_deg",
    "source_1_elevation_deg",
    "source_1_horizontal_distance",
    "source_1_distance",
    "source_2_x",
    "source_2_y",
    "source_2_z",
    "source_2_azimuth_deg",
    "source_2_elevation_deg",
    "source_2_horizontal_distance",
    "source_2_distance",
    "azimuth_difference_deg",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create fixed spatial scenarios for Libri2Mix.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--librimix-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@contextmanager
def open_csv(path, mode):
    if str(path).endswith(".gz"):
        if mode == "w":
            with open(path, "wb") as raw:
                with gzip.GzipFile(filename="",
                                   mode="wb",
                                   fileobj=raw,
                                   mtime=0) as compressed:
                    with io.TextIOWrapper(compressed,
                                          newline="",
                                          encoding="utf-8") as stream:
                        yield stream
            return
        with gzip.open(path, mode + "t", newline="",
                       encoding="utf-8") as stream:
            yield stream
        return
    with open(path, mode, newline="", encoding="utf-8") as stream:
        yield stream


def speaker_ids(mixture_id):
    parts = mixture_id.split("_")
    if len(parts) != 2:
        raise ValueError(
            f"Expected a two-speaker Libri2Mix ID, got: {mixture_id}")
    return parts[0].split("-")[0], parts[1].split("-")[0]


def scenario_seed(global_seed, version, split, mixture_id):
    value = f"{version}:{global_seed}:{split}:{mixture_id}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def angular_difference(first, second):
    difference = abs(first - second) % 360.0
    return min(difference, 360.0 - difference)


def sample_scenario(config, split, mixture_id):
    version = config["version"]
    seed = scenario_seed(config["dataset"]["seed"], version, split, mixture_id)
    rng = np.random.default_rng(seed)
    room_cfg = config["room"]
    array_cfg = config["array"]
    source_cfg = config["sources"]
    margin = float(room_cfg["boundary_margin"])
    array_radius = np.linalg.norm(
        np.asarray(array_cfg["mic_positions"], dtype=np.float64),
        axis=1,
    ).max()

    # Sample a complete valid room so every source and microphone is inside.
    for _ in range(config["dataset"]["max_sampling_attempts"]):
        room = rng.uniform(room_cfg["dimensions"]["min"],
                           room_cfg["dimensions"]["max"])
        rt60 = rng.uniform(room_cfg["rt60"]["min"], room_cfg["rt60"]["max"])
        xy_margin = margin + array_radius
        array_center = np.array([
            rng.uniform(xy_margin, room[0] - xy_margin),
            rng.uniform(xy_margin, room[1] - xy_margin),
            rng.uniform(array_cfg["height"]["min"],
                        array_cfg["height"]["max"]),
        ])
        rotation = rng.uniform(array_cfg["rotation_deg"]["min"],
                               array_cfg["rotation_deg"]["max"])

        sources = []
        for _ in range(source_cfg["num_speakers"]):
            azimuth = rng.uniform(source_cfg["azimuth_deg"]["min"],
                                  source_cfg["azimuth_deg"]["max"])
            horizontal = rng.uniform(
                source_cfg["horizontal_distance"]["min"],
                source_cfg["horizontal_distance"]["max"],
            )
            height = rng.uniform(source_cfg["height"]["min"],
                                 source_cfg["height"]["max"])
            global_angle = math.radians(azimuth + rotation)
            position = np.array([
                array_center[0] + horizontal * math.cos(global_angle),
                array_center[1] + horizontal * math.sin(global_angle),
                height,
            ])
            vertical = height - array_center[2]
            elevation = math.degrees(math.atan2(vertical, horizontal))
            distance = math.hypot(horizontal, vertical)
            sources.append(
                (position, azimuth, elevation, horizontal, distance))

        inside = all(
            np.all(item[0] >= margin) and np.all(item[0] <= room - margin)
            for item in sources)
        separation = angular_difference(sources[0][1], sources[1][1])
        if inside and separation >= source_cfg["min_azimuth_difference_deg"]:
            break
    else:
        raise RuntimeError(f"Failed to sample a valid room for {mixture_id}")

    spk1, spk2 = speaker_ids(mixture_id)
    row = {
        "version": version,
        "split": split,
        "mixture_id": mixture_id,
        "speaker_1": spk1,
        "speaker_2": spk2,
        "scenario_seed": seed,
        "room_x": room[0],
        "room_y": room[1],
        "room_z": room[2],
        "rt60": rt60,
        "array_x": array_center[0],
        "array_y": array_center[1],
        "array_z": array_center[2],
        "array_rotation_deg": rotation,
        "azimuth_difference_deg": separation,
    }
    for index, item in enumerate(sources, 1):
        position, azimuth, elevation, horizontal, distance = item
        row.update({
            f"source_{index}_x": position[0],
            f"source_{index}_y": position[1],
            f"source_{index}_z": position[2],
            f"source_{index}_azimuth_deg": azimuth,
            f"source_{index}_elevation_deg": elevation,
            f"source_{index}_horizontal_distance": horizontal,
            f"source_{index}_distance": distance,
        })
    return row


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["sources"]["num_speakers"] != 2:
        raise ValueError("This Libri2Mix recipe requires exactly two speakers")

    root = Path(args.librimix_root)
    rows = []
    for split in config["dataset"]["splits"]:
        s1_dir = root / split / "s1"
        s2_dir = root / split / "s2"
        if not s1_dir.is_dir() or not s2_dir.is_dir():
            raise FileNotFoundError(
                f"Expected Libri2Mix source directories under {root / split}")
        source_paths = sorted(s1_dir.glob("*.wav"))
        if not source_paths:
            raise RuntimeError(f"No Libri2Mix sources found in {s1_dir}")
        for source_path in source_paths:
            if not (s2_dir / source_path.name).is_file():
                raise FileNotFoundError(s2_dir / source_path.name)
            rows.append(sample_scenario(config, split, source_path.stem))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open_csv(output, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.8f}" if isinstance(value, float) else value
                for key, value in row.items()
            })
    print(f"Saved {len(rows)} fixed scenarios to {output}")


if __name__ == "__main__":
    main()
