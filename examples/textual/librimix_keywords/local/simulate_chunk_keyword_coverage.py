#!/usr/bin/env python3
# Copyright (c) 2026 Haoyu Li (haoyu.li.cs@sjtu.edu.cn)
#
# SPDX-License-Identifier: Apache-2.0

"""Measure static LibriMix chunk coverage before enabling cue rejection.

The simulator mirrors ``wesep.dataset.processor.get_random_chunk`` for static
mixtures: one shared crop for the mixture and all sources, three finite retries
for an all-zero mixture crop, and silence padding for short utterances. It
never changes audio, data manifests, or cue resources.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import multiprocessing as mp
import os
from pathlib import Path
import random

import numpy as np
import soundfile as sf

from wesep.dataset.textual_alignment import (
    AlignmentNotFoundError,
    LibriSpeechAlignmentReader,
    load_word_to_phone_ids,
)

_MAX_ZERO_RETRY = 3
_WORKER_STATE = {}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--textual-resource", type=Path, required=True)
    parser.add_argument("--primary-alignment-path", type=Path, required=True)
    parser.add_argument("--fallback-alignment-path", type=Path)
    parser.add_argument("--word2lexicon-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-seconds", nargs="+", type=float,
                        default=[3, 4, 5, 6])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-workers", type=int,
                        default=min(os.cpu_count() or 1, 32))
    return parser.parse_args()


def _load_samples(path):
    samples = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            key = sample.get("key")
            speakers = sample.get("spk")
            mix_paths = sample.get("mix", {}).get("default", [])
            if (not isinstance(key, str) or not isinstance(speakers, list)
                    or not speakers or not mix_paths):
                raise ValueError(f"Invalid sample at {path}:{line_number}")
            utterances = key.split("_")
            if len(utterances) != len(speakers):
                raise ValueError(
                    f"Source count mismatch at {path}:{line_number}: {key}")
            samples.append((line_number, sample))
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples


def _init_worker(primary_path, fallback_path, lexicon_path, resource_path):
    _WORKER_STATE["reader"] = LibriSpeechAlignmentReader(
        primary_path, fallback_path)
    _WORKER_STATE["lexicon"] = load_word_to_phone_ids(lexicon_path)
    with Path(resource_path).open(encoding="utf-8") as stream:
        _WORKER_STATE["textual_resource"] = json.load(stream)


def _is_zero_crop(nonzero_prefix, start, end):
    return nonzero_prefix[end] == nonzero_prefix[start]


def _draw_chunk(num_frames, chunk_length, nonzero_prefix, rng):
    if num_frames < chunk_length:
        return {
            "chunk_start": 0,
            "chunk_end": num_frames,
            "valid_length": num_frames,
            "zero_retry_count": 0,
            "zero_retry_exhausted": False,
        }

    chunk_start = rng.randint(0, num_frames - chunk_length)
    zero_retry_count = 0
    for _ in range(_MAX_ZERO_RETRY):
        if not _is_zero_crop(nonzero_prefix, chunk_start,
                             chunk_start + chunk_length):
            break
        zero_retry_count += 1
        chunk_start = rng.randint(0, num_frames - chunk_length)
    exhausted = _is_zero_crop(nonzero_prefix, chunk_start,
                              chunk_start + chunk_length)
    return {
        "chunk_start": chunk_start,
        "chunk_end": chunk_start + chunk_length,
        "valid_length": chunk_length,
        "zero_retry_count": zero_retry_count,
        "zero_retry_exhausted": bool(exhausted),
    }


def _alignment_for_utterance(utterance_id):
    reader = _WORKER_STATE["reader"]
    try:
        lookup = reader.get_words(utterance_id)
        return lookup, None
    except AlignmentNotFoundError:
        return None, "missing_primary_and_fallback_alignment"


def _failure_row(sample, speaker_slot, utterance_id, chunk, chunk_seconds,
                 epoch, reason, alignment_source=None):
    return {
        "mixture_uid": sample["key"],
        "speaker_slot": speaker_slot,
        "utterance_uid": utterance_id,
        "chunk_seconds": chunk_seconds,
        "virtual_epoch": epoch,
        "chunk_start": chunk["chunk_start"],
        "chunk_end": chunk["chunk_end"],
        "valid_length": chunk["valid_length"],
        "zero_retry_count": chunk["zero_retry_count"],
        "zero_retry_exhausted": chunk["zero_retry_exhausted"],
        "alignment_source": alignment_source,
        "failure_reason": reason,
    }


def _simulate_one(sample_index_and_sample, chunk_seconds, epochs, seed,
                  sample_rate):
    line_number, sample = sample_index_and_sample
    mix_path = Path(sample["mix"]["default"][0])
    waveform, actual_sample_rate = sf.read(mix_path, always_2d=True,
                                            dtype="float32")
    if actual_sample_rate != sample_rate:
        raise ValueError(
            f"Unexpected sample rate for {mix_path}: {actual_sample_rate} != "
            f"{sample_rate}")
    num_frames = waveform.shape[0]
    nonzero_prefix = np.concatenate((
        np.array([0], dtype=np.int64),
        np.cumsum(np.any(waveform != 0.0, axis=1), dtype=np.int64),
    ))
    del waveform

    stats = Counter()
    rows = []
    chunk_length = int(round(chunk_seconds * sample_rate))
    utterances = sample["key"].split("_")
    speakers = sample["spk"]
    textual_resource = _WORKER_STATE["textual_resource"]
    lexicon = _WORKER_STATE["lexicon"]

    # Cache archive lookups once per source utterance; crops differ per draw.
    alignment_data = {
        utterance_id: _alignment_for_utterance(utterance_id)
        for utterance_id in utterances
    }

    for epoch in range(epochs):
        # Per-sample streams make the simulation deterministic under changing
        # worker counts while retaining production ``random.randint`` draws.
        rng = random.Random(f"{seed}:{line_number}:{chunk_seconds}:{epoch}")
        chunk = _draw_chunk(num_frames, chunk_length, nonzero_prefix, rng)
        stats["mixture_draws"] += 1
        if chunk["zero_retry_count"]:
            stats["all_zero_mixture_crop_retries"] += chunk["zero_retry_count"]
            rows.append(_failure_row(
                sample, None, None, chunk, chunk_seconds, epoch,
                "all_zero_mixture_crop_retry"))
        if chunk["zero_retry_exhausted"]:
            stats["all_zero_mixture_crop_retry_exhausted"] += 1
            rows.append(_failure_row(
                sample, None, None, chunk, chunk_seconds, epoch,
                "all_zero_mixture_crop_retry_exhausted"))

        discard_mixture = False
        chunk_start_seconds = chunk["chunk_start"] / sample_rate
        chunk_end_seconds = chunk["chunk_end"] / sample_rate
        for speaker_slot, (utterance_id, speaker_id) in enumerate(
                zip(utterances, speakers), start=1):
            stats["target_speaker_draws"] += 1
            lookup_key = f"{sample['key']}::{speaker_id}"
            if lookup_key not in textual_resource:
                stats["missing_textual_resource"] += 1
                rows.append(_failure_row(
                    sample, speaker_slot, utterance_id, chunk, chunk_seconds,
                    epoch, "missing_textual_resource"))
                discard_mixture = True
                continue

            lookup, alignment_error = alignment_data[utterance_id]
            if alignment_error is not None:
                stats["missing_primary_alignment"] += 1
                stats["missing_fallback_alignment"] += 1
                rows.append(_failure_row(
                    sample, speaker_slot, utterance_id, chunk, chunk_seconds,
                    epoch, alignment_error))
                discard_mixture = True
                continue

            stats[f"alignment_{lookup.source}_used"] += 1
            if not lookup.primary_available:
                stats["missing_primary_alignment"] += 1

            complete_words = [
                word for word in lookup.words
                if word.start >= chunk_start_seconds - 1e-4
                and word.end <= chunk_end_seconds + 1e-4
            ]
            if not complete_words:
                stats["no_complete_aligned_word"] += 1
                rows.append(_failure_row(
                    sample, speaker_slot, utterance_id, chunk, chunk_seconds,
                    epoch, "no_complete_aligned_word", lookup.source))
                discard_mixture = True
                continue
            missing_word = next(
                (word.text for word in complete_words
                 if word.text not in lexicon), None)
            if missing_word is not None:
                stats["missing_lexicon_entry"] += 1
                rows.append(_failure_row(
                    sample, speaker_slot, utterance_id, chunk, chunk_seconds,
                    epoch, f"missing_lexicon:{missing_word}", lookup.source))
                discard_mixture = True

        if discard_mixture:
            stats["discarded_mixture_draws"] += 1

    return stats, rows


def _simulate_worker(task):
    sample, chunk_seconds, epochs, seed, sample_rate = task
    return _simulate_one(sample, chunk_seconds, epochs, seed, sample_rate)


def _format_percentage(numerator, denominator):
    if not denominator:
        return "0.000%"
    return f"{100.0 * numerator / denominator:.3f}%"


def _write_summary_markdown(output_path, summary, args):
    lines = [
        "# Static LibriMix chunk keyword coverage simulation",
        "",
        f"- samples: `{args.samples}`",
        f"- virtual epochs: `{args.epochs}`",
        f"- seed: `{args.seed}`",
        f"- primary alignment: `{args.primary_alignment_path}`",
        f"- fallback alignment: `{args.fallback_alignment_path}`",
        "",
        "| Chunk | Mixture draws | Target draws | Discarded mixtures | "
        "Discard rate | No complete word | Missing primary | Missing fallback | "
        "Fallback used | Zero retries | Exhausted zero retries |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for chunk_key, stats in summary["by_chunk_seconds"].items():
        mixture_draws = stats["mixture_draws"]
        lines.append(
            "| {chunk} | {mix} | {target} | {discarded} | {rate} | {no_word} | "
            "{missing_primary} | {missing_fallback} | {fallback} | {retries} | "
            "{exhausted} |".format(
                chunk=chunk_key,
                mix=mixture_draws,
                target=stats["target_speaker_draws"],
                discarded=stats["discarded_mixture_draws"],
                rate=_format_percentage(stats["discarded_mixture_draws"],
                                        mixture_draws),
                no_word=stats["no_complete_aligned_word"],
                missing_primary=stats["missing_primary_alignment"],
                missing_fallback=stats["missing_fallback_alignment"],
                fallback=stats["alignment_fallback_used"],
                retries=stats["all_zero_mixture_crop_retries"],
                exhausted=stats["all_zero_mixture_crop_retry_exhausted"],
            ))
    lines.extend([
        "",
        "Detailed rows are failure and all-zero-retry events only; successful "
        "draws are represented by the aggregate counts above.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if any(seconds <= 0 for seconds in args.chunk_seconds):
        raise ValueError("--chunk-seconds values must be positive")
    for path in (args.samples, args.textual_resource,
                 args.primary_alignment_path, args.word2lexicon_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.fallback_alignment_path and not args.fallback_alignment_path.exists():
        raise FileNotFoundError(args.fallback_alignment_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = args.output_dir / "detailed.jsonl"
    summary_path = args.output_dir / "summary.json"
    markdown_path = args.output_dir / "summary.md"
    if any(path.exists() for path in (detailed_path, summary_path, markdown_path)):
        raise FileExistsError(
            f"Simulation output already exists in {args.output_dir}; choose a "
            "new --output-dir to avoid mixing results.")

    samples = _load_samples(args.samples)
    summary = {"by_chunk_seconds": {}}
    initializer_args = (
        str(args.primary_alignment_path),
        str(args.fallback_alignment_path) if args.fallback_alignment_path else None,
        str(args.word2lexicon_path),
        str(args.textual_resource),
    )
    with detailed_path.open("w", encoding="utf-8") as detailed_stream:
        for chunk_seconds in args.chunk_seconds:
            aggregate = Counter()
            tasks = [
                (sample, chunk_seconds, args.epochs, args.seed, args.sample_rate)
                for sample in samples
            ]
            with mp.Pool(
                processes=args.num_workers,
                initializer=_init_worker,
                initargs=initializer_args,
            ) as pool:
                for stats, rows in pool.imap_unordered(_simulate_worker, tasks,
                                                       chunksize=8):
                    aggregate.update(stats)
                    for row in rows:
                        detailed_stream.write(json.dumps(row, sort_keys=True) + "\n")
            summary["by_chunk_seconds"][str(chunk_seconds)] = {
                key: aggregate[key]
                for key in (
                    "mixture_draws", "target_speaker_draws",
                    "all_zero_mixture_crop_retries",
                    "all_zero_mixture_crop_retry_exhausted",
                    "alignment_primary_used", "alignment_fallback_used",
                    "missing_primary_alignment", "missing_fallback_alignment",
                    "missing_textual_resource", "no_complete_aligned_word",
                    "missing_lexicon_entry", "discarded_mixture_draws",
                )
            }
            print(
                f"chunk={chunk_seconds}s mixtures={aggregate['mixture_draws']} "
                f"discarded={aggregate['discarded_mixture_draws']}",
                flush=True,
            )

    summary["config"] = {
        "samples": str(args.samples),
        "textual_resource": str(args.textual_resource),
        "primary_alignment_path": str(args.primary_alignment_path),
        "fallback_alignment_path": (str(args.fallback_alignment_path)
                                    if args.fallback_alignment_path else None),
        "word2lexicon_path": str(args.word2lexicon_path),
        "chunk_seconds": args.chunk_seconds,
        "epochs": args.epochs,
        "seed": args.seed,
        "sample_rate": args.sample_rate,
        "num_workers": args.num_workers,
        "max_zero_retry": _MAX_ZERO_RETRY,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    _write_summary_markdown(markdown_path, summary, args)
    print(f"Wrote {summary_path}, {markdown_path}, and {detailed_path}")


if __name__ == "__main__":
    main()
