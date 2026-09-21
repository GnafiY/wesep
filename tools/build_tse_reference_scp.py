#!/usr/bin/env python3
# Copyright 2026 Haoyu Li (haoyu.li.cs@sjtu.edu.cn)

"""Build target-expanded TSE reference SCPs from a samples.jsonl manifest.

The TSE inference frontend expands each mixture once per source and writes
output keys in the form ``s{source_slot}/{mixture_key}.wav``. ``score.sh``
expects its reference SCP to use the same keys. This tool derives those
references from the common samples.jsonl schema without changing the source
manifest or audio files.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, help="Input samples.jsonl")
    parser.add_argument(
        "--output",
        required=True,
        help="Output target-expanded reference SCP, normally single.wav.scp",
    )
    parser.add_argument(
        "--inference-scp",
        help=("Optional enhanced-output SCP. When given, require its keys to "
              "match the generated reference keys exactly."),
    )
    parser.add_argument(
        "--check-source-files",
        action="store_true",
        help="Fail when a source waveform referenced by samples.jsonl is absent.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {error.msg}") from error
            if not isinstance(sample, dict):
                raise ValueError(
                    f"Expected an object in {path}:{line_number}, got "
                    f"{type(sample).__name__}")
            yield line_number, sample


def build_reference_entries(
    samples_path: Path,
    check_source_files: bool,
) -> list[tuple[str, str]]:
    entries: dict[str, str] = {}
    for line_number, sample in read_jsonl(samples_path):
        key = sample.get("key")
        speakers = sample.get("spk")
        sources = sample.get("src")
        location = f"{samples_path}:{line_number}"

        if not isinstance(key, str) or not key:
            raise ValueError(f"Missing non-empty string 'key' at {location}")
        if any(character.isspace() for character in key):
            raise ValueError(
                f"Whitespace is unsupported in sample key at {location}: {key!r}")
        if not isinstance(speakers, list) or not speakers:
            raise ValueError(f"Missing non-empty speaker list 'spk' at {location}")
        if not isinstance(sources, dict):
            raise ValueError(f"Missing source mapping 'src' at {location}")

        for source_slot, speaker_id in enumerate(speakers, start=1):
            if not isinstance(speaker_id, str) or not speaker_id:
                raise ValueError(
                    f"Invalid speaker ID for slot {source_slot} at {location}")
            source_paths = sources.get(speaker_id)
            if not isinstance(source_paths, list) or not source_paths:
                raise ValueError(
                    f"Missing source path for speaker {speaker_id!r} at {location}")
            source_path = source_paths[0]
            if not isinstance(source_path, str) or not source_path:
                raise ValueError(
                    f"Invalid first source path for speaker {speaker_id!r} at {location}")
            if any(character.isspace() for character in source_path):
                raise ValueError(
                    f"Whitespace is unsupported in source path at {location}: "
                    f"{source_path!r}")
            if check_source_files and not Path(source_path).is_file():
                raise FileNotFoundError(
                    f"Source waveform for {key}, slot {source_slot} is absent: "
                    f"{source_path}")

            target_key = f"s{source_slot}/{key}.wav"
            previous = entries.setdefault(target_key, source_path)
            if previous != source_path:
                raise ValueError(
                    f"Conflicting source paths for target key {target_key!r}: "
                    f"{previous!r} and {source_path!r}")

    if not entries:
        raise ValueError(f"No samples found in {samples_path}")
    return sorted(entries.items())


def read_scp_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(
                    f"Expected '<key> <path>' in {path}:{line_number}")
            key = fields[0]
            if key in keys:
                raise ValueError(f"Duplicate key {key!r} in {path}:{line_number}")
            keys.add(key)
    if not keys:
        raise ValueError(f"No entries found in {path}")
    return keys


def verify_inference_keys(entries: list[tuple[str, str]], inference_scp: Path) -> None:
    if not inference_scp.is_file():
        raise FileNotFoundError(f"Enhanced-output SCP is absent: {inference_scp}")
    reference_keys = {key for key, _ in entries}
    inference_keys = read_scp_keys(inference_scp)
    missing = sorted(reference_keys - inference_keys)
    extra = sorted(inference_keys - reference_keys)
    if missing or extra:
        details = []
        if missing:
            details.append(
                f"missing {len(missing)} inference key(s), e.g. {missing[:3]}")
        if extra:
            details.append(
                f"unexpected {len(extra)} inference key(s), e.g. {extra[:3]}")
        raise ValueError(
            f"Reference and enhanced SCP key sets differ: {'; '.join(details)}")


def write_scp(entries: list[tuple[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        for key, source_path in entries:
            stream.write(f"{key} {source_path}\n")
    os.replace(temporary_path, output_path)


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples)
    output_path = Path(args.output)
    if not samples_path.is_file():
        raise FileNotFoundError(f"Samples manifest is absent: {samples_path}")

    entries = build_reference_entries(samples_path, args.check_source_files)
    if args.inference_scp:
        verify_inference_keys(entries, Path(args.inference_scp))
    write_scp(entries, output_path)
    print(f"Wrote {len(entries)} target references to {output_path}")


if __name__ == "__main__":
    main()
