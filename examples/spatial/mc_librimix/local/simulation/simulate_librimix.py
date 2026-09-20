#!/usr/bin/env python3
"""Generate early and fully reverberant multichannel Libri2Mix audio."""

import argparse
import csv
import gzip
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy import signal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spatialize fixed Libri2Mix s1/s2 waveforms with gpuRIR.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--librimix-root", required=True)
    parser.add_argument("--scenario-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing waveforms after scenario changes.")
    return parser.parse_args()


def read_scenarios(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def rotate_array(local_positions, center, rotation_deg):
    angle = math.radians(rotation_deg)
    rotation = np.array([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return local_positions @ rotation.T + center


def load_source(path, sample_rate):
    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz, got {rate} Hz: {path}")
    if waveform.shape[1] != 1:
        raise ValueError(f"Expected a mono Libri2Mix source: {path}")
    return waveform[:, 0]


def convolve_sources(sources, rirs, output_length):
    images = np.empty((len(sources), output_length, rirs.shape[1]),
                      dtype=np.float32)
    for source_index, source in enumerate(sources):
        for channel in range(rirs.shape[1]):
            images[source_index, :, channel] = signal.fftconvolve(
                source,
                rirs[source_index, channel],
                mode="full",
            )[:output_length]
    return images


def early_rirs(full_rirs, sources, microphones, config):
    sample_rate = config["audio"]["sample_rate"]
    rir_cfg = config["rir"]
    early_samples = round(rir_cfg["early_window_ms"] * sample_rate / 1000)
    early = full_rirs.copy()
    for source_index, source in enumerate(sources):
        for channel, microphone in enumerate(microphones):
            if rir_cfg["preserve_time_of_flight"]:
                distance = np.linalg.norm(source - microphone)
                arrival = round(distance / rir_cfg["sound_speed"] *
                                sample_rate)
            else:
                arrival = 0
            early[source_index, channel, arrival + early_samples:] = 0.0
    return early


def expected_paths(base_dir, name, outputs):
    paths = []
    if outputs["early_source"]:
        paths.extend(
            [base_dir / "s1_early" / name, base_dir / "s2_early" / name])
    if outputs["early_mixture"]:
        paths.append(base_dir / "mix_early" / name)
    if outputs["reverb_source"]:
        paths.extend(
            [base_dir / "s1_reverb" / name, base_dir / "s2_reverb" / name])
    if outputs["reverb_mixture"]:
        paths.append(base_dir / "mix_reverb" / name)
    return paths


def main():
    args = parse_args()
    try:
        import gpuRIR
    except ImportError as error:
        raise RuntimeError(
            "gpuRIR is required to generate the spatial dataset") from error

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    sample_rate = config["audio"]["sample_rate"]
    local_mics = np.asarray(config["array"]["mic_positions"], dtype=np.float64)
    outputs = config["outputs"]
    input_root = Path(args.librimix_root)
    output_root = Path(args.output_root)

    scenarios = list(read_scenarios(args.scenario_manifest))
    if args.split != "all":
        scenarios = [row for row in scenarios if row["split"] == args.split]

    for index, row in enumerate(scenarios, 1):
        split = row["split"]
        mixture_id = row["mixture_id"]
        filename = f"{mixture_id}.wav"
        base_dir = output_root / split
        paths = expected_paths(base_dir, filename, outputs)
        if not args.overwrite and paths and all(path.is_file()
                                                for path in paths):
            continue
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)

        # Read the gain-adjusted and length-aligned Libri2Mix sources.
        first = load_source(input_root / split / "s1" / filename, sample_rate)
        second = load_source(input_root / split / "s2" / filename, sample_rate)
        if first.shape != second.shape:
            raise ValueError(
                f"Libri2Mix sources are not aligned: {mixture_id}")
        source_audio = [first, second]

        room = np.array([row["room_x"], row["room_y"], row["room_z"]],
                        dtype=np.float64)
        center = np.array([row["array_x"], row["array_y"], row["array_z"]],
                          dtype=np.float64)
        microphones = rotate_array(local_mics, center,
                                   float(row["array_rotation_deg"]))
        sources = np.array([
            [row["source_1_x"], row["source_1_y"], row["source_1_z"]],
            [row["source_2_x"], row["source_2_y"], row["source_2_z"]],
        ],
                           dtype=np.float64)
        rt60 = float(row["rt60"])

        # Generate one full RIR and derive the early response in memory.
        beta = gpuRIR.beta_SabineEstimation(room, rt60)
        image_order = gpuRIR.t2n(rt60, room)
        full_rirs = gpuRIR.simulateRIR(room, beta, sources, microphones,
                                       image_order, rt60, sample_rate)
        full_rirs = np.asarray(full_rirs, dtype=np.float32)
        early = early_rirs(full_rirs, sources, microphones, config)

        length = len(first)
        early_images = convolve_sources(source_audio, early, length)
        reverb_images = convolve_sources(source_audio, full_rirs, length)

        wav_outputs = []
        if outputs["early_source"]:
            wav_outputs.extend([
                (base_dir / "s1_early" / filename, early_images[0]),
                (base_dir / "s2_early" / filename, early_images[1]),
            ])
        if outputs["early_mixture"]:
            wav_outputs.append(
                (base_dir / "mix_early" / filename, early_images.sum(axis=0)))
        if outputs["reverb_source"]:
            wav_outputs.extend([
                (base_dir / "s1_reverb" / filename, reverb_images[0]),
                (base_dir / "s2_reverb" / filename, reverb_images[1]),
            ])
        if outputs["reverb_mixture"]:
            wav_outputs.append((base_dir / "mix_reverb" / filename,
                                reverb_images.sum(axis=0)))

        if not wav_outputs:
            raise ValueError("At least one waveform output must be enabled")
        peak = max(float(np.max(np.abs(audio))) for _, audio in wav_outputs)
        scale = 1.0 / peak if peak > 1.0 else 1.0
        for path, audio in wav_outputs:
            sf.write(path, audio * scale, sample_rate, subtype="PCM_16")
        if index % 100 == 0 or index == len(scenarios):
            print(f"Generated {index}/{len(scenarios)} spatial mixtures")


if __name__ == "__main__":
    main()
