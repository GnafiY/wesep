#!/usr/bin/env python3
"""Build word-aligned DAE phoneme cues from LibriSpeech transcripts."""

import argparse
import json
import random
from pathlib import Path

from wesep.modules.textual.kce import DAEPhonemeTokenizer


def load_transcripts(root):
    """Collect LibriSpeech utterance transcripts from *.trans.txt files."""
    transcripts = {}
    for path in sorted(Path(root).rglob("*.trans.txt")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                fields = line.strip().split(maxsplit=1)
                if len(fields) != 2:
                    continue
                key, text = fields
                if key in transcripts and transcripts[key] != text:
                    raise RuntimeError(f"Conflicting transcript for {key}")
                transcripts[key] = text
    if not transcripts:
        raise RuntimeError(f"No LibriSpeech transcripts found under {root}")
    return transcripts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--librispeech-root", required=True)
    parser.add_argument("--phoneme-map", required=True)
    parser.add_argument("--lexicon", default=None)
    parser.add_argument("--fixed-words", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    transcripts = load_transcripts(args.librispeech_root)
    tokenizer = DAEPhonemeTokenizer(args.phoneme_map, args.lexicon)
    generator = random.Random(args.seed)
    cues = {}

    # Each Libri2Mix key lists source utterances in the same order as speakers.
    with open(args.samples, encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            utterances = sample["key"].split("_")
            speakers = sample["spk"]
            if len(utterances) != len(speakers):
                raise RuntimeError(
                    f"Source count mismatch for mixture {sample['key']}")

            for utterance, speaker in zip(utterances, speakers):
                if utterance not in transcripts:
                    raise KeyError(f"Transcript not found: {utterance}")
                normalized, words, labels = tokenizer.encode_words(
                    transcripts[utterance])
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
