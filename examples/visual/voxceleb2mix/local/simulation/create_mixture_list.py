#!/usr/bin/env python3
"""Create a deterministic MuSE-style VoxCeleb2 mixture manifest."""

import argparse
import csv
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Utterance:
    split: str
    speaker: str
    clip: str
    duration: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a fixed two-speaker VoxCeleb2 mixture list.")
    parser.add_argument(
        "--audio-root",
        required=True,
        help="Root containing train/test/<speaker>/<video>/<clip>.wav.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-speakers", type=int, default=800)
    parser.add_argument("--val-utts-per-speaker", type=int, default=12)
    parser.add_argument("--train-utts-per-speaker", type=int, default=50)
    parser.add_argument("--train-mixtures", type=int, default=20000)
    parser.add_argument("--val-mixtures", type=int, default=5000)
    parser.add_argument("--test-mixtures", type=int, default=3000)
    parser.add_argument("--min-duration", type=float, default=4.0)
    parser.add_argument("--mix-db", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def scan_split(root, split, sample_rate, min_duration):
    """Read sorted WAV metadata without loading complete waveforms."""
    utterances = []
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)

    for path in sorted(split_root.glob("*/*/*.wav")):
        relative = path.relative_to(split_root)
        speaker, video, filename = relative.parts
        with wave.open(str(path), "rb") as stream:
            if stream.getframerate() != sample_rate:
                raise ValueError(f"Expected {sample_rate} Hz, got "
                                 f"{stream.getframerate()} Hz: {path}")
            if stream.getnchannels() != 1:
                raise ValueError(f"Expected mono audio: {path}")
            duration = stream.getnframes() / sample_rate
        if duration >= min_duration:
            utterances.append(
                Utterance(
                    split=split,
                    speaker=speaker,
                    clip=f"{video}/{Path(filename).stem}",
                    duration=duration,
                ))
    if not utterances:
        raise RuntimeError(f"No eligible WAV files found in {split_root}")
    return utterances


def split_train_validation(utterances, args):
    """Select the most frequent speakers and assign fixed utterance ranges."""
    counts = Counter(item.speaker for item in utterances)
    speakers = sorted(counts, key=lambda speaker: (-counts[speaker], speaker))
    selected = set(speakers[:args.num_speakers])
    grouped = {speaker: [] for speaker in selected}
    for item in utterances:
        if item.speaker in selected:
            grouped[item.speaker].append(item)

    train, validation = [], []
    for speaker in sorted(grouped):
        items = grouped[speaker]
        val_end = args.val_utts_per_speaker
        train_end = val_end + args.train_utts_per_speaker
        validation.extend(items[:val_end])
        train.extend(items[val_end:train_end])
    if len(grouped) < args.num_speakers:
        raise RuntimeError(
            f"Requested {args.num_speakers} speakers, found {len(grouped)}")
    return train, validation


def sample_pair(pool, rng):
    """Draw two utterances from different speakers."""
    first = pool[int(rng.randint(len(pool)))]
    second = pool[int(rng.randint(len(pool)))]
    while second.speaker == first.speaker:
        second = pool[int(rng.randint(len(pool)))]
    return first, second


def pop_random(pool, rng):
    """Remove one uniformly sampled item without shifting the full list."""
    index = int(rng.randint(len(pool)))
    pool[index], pool[-1] = pool[-1], pool[index]
    return pool.pop()


def sample_rows(split, pool, count, mix_db, rng, replace):
    """Follow MuSE's no-replacement pass before optional repeated sampling."""
    rows = []
    remaining = list(pool)
    if not replace:
        speaker_counts = Counter(item.speaker for item in remaining)
        while len(remaining) >= 2 and len(rows) < count:
            first = pop_random(remaining, rng)
            speaker_counts[first.speaker] -= 1
            if len(remaining) == speaker_counts[first.speaker]:
                break
            second = pop_random(remaining, rng)
            while second.speaker == first.speaker:
                remaining.append(second)
                second = pop_random(remaining, rng)
            speaker_counts[second.speaker] -= 1
            gain = rng.uniform(-mix_db, mix_db)
            rows.append(mixture_row(split, first, second, gain))

    while len(rows) < count:
        first, second = sample_pair(pool, rng)
        gain = rng.uniform(-mix_db, mix_db)
        rows.append(mixture_row(split, first, second, gain))
    return rows


def mixture_row(split, first, second, gain_db):
    duration = min(first.duration, second.duration)
    return [
        split,
        first.split,
        first.speaker,
        first.clip,
        0.0,
        second.split,
        second.speaker,
        second.clip,
        gain_db,
        duration,
    ]


def main():
    args = parse_args()
    root = Path(args.audio_root)
    rng = np.random.RandomState(args.seed)

    # Build the MuSE 800-speaker train/validation partition.
    pretrain = scan_split(root, "train", args.sample_rate, args.min_duration)
    train, validation = split_train_validation(pretrain, args)
    test = scan_split(root, "test", args.sample_rate, args.min_duration)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("train", train, args.train_mixtures, False),
        ("val", validation, args.val_mixtures, False),
        ("test", test, args.test_mixtures, True),
    ]
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for split, pool, count, replace in jobs:
            if len({item.speaker for item in pool}) < 2:
                raise RuntimeError(
                    f"The {split} pool must contain at least two speakers")
            writer.writerows(
                sample_rows(split, pool, count, args.mix_db, rng, replace))

    print(f"Saved {sum(item[2] for item in jobs)} mixtures to {output}: "
          f"train={len(train)}, val={len(validation)}, test={len(test)} "
          "eligible utterances")


if __name__ == "__main__":
    main()
