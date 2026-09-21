#!/usr/bin/env python3
"""Build DAE phoneme cues from Nemo Libri2Mix source transcriptions."""

import argparse
import json
import random
from pathlib import Path, PurePath

from wesep.modules.textual.kce import DAEPhonemeTokenizer


def _parse_source_index(audio_filepath: str) -> int:
    """Read the Libri2Mix source index from a Nemo audio path.

    Expected records refer to paths such as
    ``.../mix_clean/../s1/<mixture_uid>.wav``.  Do not resolve the path: the
    hosted manifest need not use this machine's absolute dataset root.
    """
    for component in reversed(PurePath(audio_filepath).parts[:-1]):
        if component.startswith("s") and component[1:].isdigit():
            source_index = int(component[1:])
            if source_index >= 1:
                return source_index
    raise ValueError(
        "Nemo transcript audio_filepath does not contain a source directory "
        f"such as s1 or s2: {audio_filepath}")


def load_nemo_transcripts(path: Path) -> dict[tuple[str, str, int], str]:
    """Index ``pred_text`` by (Libri2Mix subset, mixture UID, source index)."""
    transcripts = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                item = json.loads(line)
                subset = item["subset"]
                audio_filepath = item["audio_filepath"]
                text = item["pred_text"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"Invalid Nemo transcript at {path}:{line_number}") from error
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Empty pred_text at {path}:{line_number}")

            mixture_uid = PurePath(audio_filepath).stem
            source_index = _parse_source_index(audio_filepath)
            key = (subset, mixture_uid, source_index)
            if key in transcripts and transcripts[key] != text:
                raise ValueError(f"Conflicting Nemo transcript for {key}")
            transcripts[key] = text

    if not transcripts:
        raise RuntimeError(f"No Nemo transcripts found in {path}")
    return transcripts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--transcript-nemo-jsonl", required=True)
    parser.add_argument("--phoneme-map", required=True)
    parser.add_argument("--lexicon", default=None)
    parser.add_argument("--fixed-words", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    transcript_path = Path(args.transcript_nemo_jsonl)
    transcripts = load_nemo_transcripts(transcript_path)
    tokenizer = DAEPhonemeTokenizer(args.phoneme_map, args.lexicon)
    generator = random.Random(args.seed)
    cues = {}

    # Every sample key has source utterances ordered like s1, s2, ... .  Nemo
    # records identify the same source by the mixture UID and s-index.
    with open(args.samples, encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            utterances = sample["key"].split("_")
            speakers = sample["spk"]
            if len(utterances) != len(speakers):
                raise RuntimeError(
                    f"Source count mismatch for mixture {sample['key']}")

            for source_index, (utterance, speaker) in enumerate(
                    zip(utterances, speakers), start=1):
                transcript_key = (args.subset, sample["key"], source_index)
                if transcript_key not in transcripts:
                    raise KeyError(
                        "Nemo transcript not found for "
                        f"subset={args.subset}, mixture={sample['key']}, "
                        f"source=s{source_index}, utterance={utterance}")
                normalized, words, labels = tokenizer.encode_words(
                    transcripts[transcript_key])
                item = {
                    "normalized_text": normalized,
                    "words": words,
                    "phn_label": labels,
                }
                if args.fixed_words is not None:
                    count = min(args.fixed_words, len(words))
                    start = generator.randint(0, len(words) - count)
                    item["kw_candidate"] = list(range(start, start + count))
                cues[f"{sample['key']}::{speaker}"] = item

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(cues, stream, ensure_ascii=False, separators=(",", ":"))
    print(f"Saved {len(cues)} textual cues to {output}")


if __name__ == "__main__":
    main()
