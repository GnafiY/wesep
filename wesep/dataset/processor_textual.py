# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#               2026 Haoyu Li (haoyu.li.cs@sjtu.edu.cn)
#
# SPDX-License-Identifier: Apache-2.0

"""Textual cue processors for the modular TSE dataset pipeline."""

from dataclasses import dataclass
import logging
from pathlib import Path
import random
import re

import numpy as np

from wesep.utils.file_utils import load_json


@dataclass(frozen=True)
class AlignedWord:
    """One lower-cased word with a half-open time interval in seconds."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class AlignmentLookup:
    """Word alignments from the merged, canonical resource directory."""

    words: tuple[AlignedWord, ...]


class AlignmentNotFoundError(KeyError):
    """Raised when an utterance is absent from the merged alignment resource."""

    def __init__(self, utterance_id):
        self.utterance_id = utterance_id
        super().__init__(f"No alignment for {utterance_id}")


class _AlignmentDirectory:
    """Read chapter records from the canonical extracted alignment directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise NotADirectoryError(
                f"Alignment resource must be an extracted directory: {self.path}")
        self._members = {}
        for member in self.path.rglob("*.alignment.txt"):
            chapter_id = member.name.removesuffix(".alignment.txt")
            if chapter_id in self._members:
                raise ValueError(
                    f"Duplicate chapter alignment {chapter_id} under {self.path}")
            self._members[chapter_id] = member
        if not self._members:
            raise ValueError(f"No alignment files found under {self.path}")
        self._chapter_cache = {}

    @staticmethod
    def _parse_records(raw_text: str) -> dict[str, tuple[AlignedWord, ...]]:
        records = {}
        for raw_line in raw_text.splitlines():
            fields = raw_line.replace('"', "").split()
            if len(fields) != 3:
                raise ValueError(f"Malformed alignment record: {raw_line}")
            utterance_id, raw_words, raw_ends = fields
            # Official MFA records timestamp every comma position, including
            # empty non-word intervals. Qwen-corrected records retain empty
            # leading/trailing commas but timestamp only spoken words.
            tokens = raw_words.split(",")
            nonempty_tokens = [token for token in tokens if token]
            ends = [float(value) for value in raw_ends.split(",")]
            if len(ends) == len(tokens):
                timed_tokens = tokens
            elif len(ends) == len(nonempty_tokens):
                timed_tokens = nonempty_tokens
            else:
                raise ValueError(
                    f"Malformed alignment record for {utterance_id}: "
                    f"{len(tokens)} comma tokens, {len(nonempty_tokens)} words, "
                    f"but {len(ends)} end times")

            previous_end = 0.0
            words = []
            for token, end in zip(timed_tokens, ends):
                if end < previous_end:
                    raise ValueError(
                        f"Non-monotonic alignment end time for {utterance_id}")
                if token:
                    words.append(AlignedWord(token.lower(), previous_end, end))
                previous_end = end
            records[utterance_id] = tuple(words)
        return records

    def get(self, utterance_id: str) -> tuple[AlignedWord, ...] | None:
        chapter_id = "-".join(utterance_id.split("-")[:2])
        if chapter_id not in self._chapter_cache:
            member = self._members.get(chapter_id)
            self._chapter_cache[chapter_id] = (
                self._parse_records(member.read_text(encoding="utf-8"))
                if member is not None else {})
        return self._chapter_cache[chapter_id].get(utterance_id)


class LibriSpeechAlignmentReader:
    """Look up word alignments in the merged extracted resource directory."""

    def __init__(self, alignment_path: str | Path):
        self._alignment = _AlignmentDirectory(alignment_path)

    def get_words(self, utterance_id: str) -> AlignmentLookup:
        words = self._alignment.get(utterance_id)
        if words is None:
            raise AlignmentNotFoundError(utterance_id)
        return AlignmentLookup(words=words)

    def close(self) -> None:
        # Kept for the worker-generator lifecycle used by sample_textual_cue.
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def load_word_to_phone_ids(path: str | Path) -> dict[str, tuple[int, ...]]:
    """Load the DAE word-to-phoneme-ID lexicon used by the KCE tokenizer."""

    lexicon = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(
                    f"Malformed lexicon entry in {path}:{line_number}: {line!r}")
            word, raw_phone_ids = fields[0].lower(), fields[1:]
            try:
                lexicon[word] = tuple(int(phone_id) for phone_id in raw_phone_ids)
            except ValueError as error:
                raise ValueError(
                    f"Non-integer phoneme ID in {path}:{line_number}") from error
    if not lexicon:
        raise ValueError(f"No lexicon entries found in {path}")
    return lexicon


# Resources are loaded once in each DataLoader worker process.
_TEXTUAL_RESOURCE_CACHE = {}
_WORD_TO_PHONE_CACHE = {}


class ChunkKeywordSelectionError(RuntimeError):
    """A configured chunk has no usable textual keyword for one source slot."""

    def __init__(self, mixture_key, speaker_slot, utterance_id, reason):
        self.mixture_key = mixture_key
        self.speaker_slot = speaker_slot
        self.utterance_id = utterance_id
        self.reason = reason
        super().__init__(
            f"Chunk keyword unavailable for mixture={mixture_key}, "
            f"slot={speaker_slot}, utterance={utterance_id}: {reason}")


def _speaker_slots(sample):
    slots = [key for key in sample if re.fullmatch(r"spk\d+", key)]
    return sorted(slots, key=lambda key: int(key[3:]))


def _lookup_key(sample, speaker_slot, key_field):
    if key_field == "spk_id":
        return sample[speaker_slot]
    if key_field == "mix_spk_id":
        mixture_key = sample.get(f"source_key_{speaker_slot}",
                                 sample.get("key"))
        if mixture_key is None:
            raise KeyError("sample missing 'key' for mix_spk_id cue")
        return f"{mixture_key}::{sample[speaker_slot]}"
    raise ValueError(f"Unsupported key_field for textual cue: {key_field}")


def _utterance_id_for_slot(sample, speaker_slot):
    source_key = sample.get(f"source_key_{speaker_slot}")
    if source_key is not None:
        return source_key

    mixture_key = sample.get("key")
    if not isinstance(mixture_key, str):
        raise KeyError("sample missing mixture key")
    utterances = mixture_key.split("_")
    source_index = int(speaker_slot[3:]) - 1
    if source_index >= len(utterances):
        raise KeyError(
            f"No utterance ID for {speaker_slot} in mixture {mixture_key}")
    return utterances[source_index]


def _validate_word_labels(labels):
    if not isinstance(labels, list) or not labels:
        raise ValueError("DAE keyword cue requires non-empty 'phn_label'")
    if any(not isinstance(word, list) or not word for word in labels):
        raise ValueError("Each DAE keyword word requires phoneme IDs")


def _select_random_word_labels(labels, word_range):
    _validate_word_labels(labels)
    if (not isinstance(word_range, (list, tuple)) or len(word_range) != 2
            or not all(isinstance(value, int) for value in word_range)
            or word_range[0] <= 0 or word_range[1] < word_range[0]):
        raise ValueError("DAE random selection requires a valid 'word_range'")
    word_count = min(random.randint(*word_range), len(labels))
    start = random.randint(0, len(labels) - word_count)
    return labels[start:start + word_count]


def _read_dae_keyword_phoneme(item, selection):
    """Select a regular whole-utterance DAE keyword phoneme sequence."""
    labels = item.get("phn_label") if isinstance(item, dict) else None
    selection_type = selection.get("type")
    if selection_type == "random":
        selected = _select_random_word_labels(
            labels, selection.get("word_range", [2, 6]))
    elif selection_type == "fixed":
        _validate_word_labels(labels)
        candidates = item.get("kw_candidate")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(
                "DAE fixed selection requires non-empty 'kw_candidate'")
        if any(not isinstance(index, int) or index < 0 or index >= len(labels)
               for index in candidates):
            raise IndexError("DAE keyword candidate is outside 'phn_label'")
        selected = [labels[index] for index in candidates]
    else:
        raise ValueError(f"Unsupported DAE keyword selection: {selection_type}")

    phonemes = [phone_id for word in selected for phone_id in word]
    if not phonemes:
        raise ValueError("DAE keyword selection produced no phoneme IDs")
    return np.asarray(phonemes, dtype=np.int64)


def _chunk_selection_options(selection):
    options = selection.get("chunk_aware")
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError("selection.chunk_aware must be a mapping")
    if selection.get("type") != "random":
        raise ValueError("Chunk-aware textual selection requires type='random'")

    required = ("alignment_path", "word2lexicon_path")
    missing = [key for key in required if not options.get(key)]
    if missing:
        raise ValueError(
            "Chunk-aware textual selection requires " + ", ".join(missing))
    policy = options.get("no_word_policy")
    if policy not in {"discard", "error"}:
        raise ValueError(
            "Chunk-aware textual selection requires "
            "no_word_policy='discard' or 'error'")
    tolerance = options.get("tolerance", 1e-4)
    if not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise ValueError("Chunk-aware textual tolerance must be non-negative")
    return {
        "alignment_path": options["alignment_path"],
        "word2lexicon_path": options["word2lexicon_path"],
        "no_word_policy": policy,
        "tolerance": float(tolerance),
    }


def _get_word_to_phone_ids(path):
    if path not in _WORD_TO_PHONE_CACHE:
        _WORD_TO_PHONE_CACHE[path] = load_word_to_phone_ids(path)
    return _WORD_TO_PHONE_CACHE[path]


def _complete_chunk_word_labels(
    sample,
    speaker_slot,
    alignment_reader,
    word_to_phone_ids,
    tolerance,
):
    utterance_id = _utterance_id_for_slot(sample, speaker_slot)
    sample_rate = sample.get("sample_rate")
    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
        raise ValueError("Chunk-aware textual selection requires sample_rate")
    if "chunk_start" not in sample or "chunk_end" not in sample:
        raise ValueError(
            "Chunk-aware textual selection requires exact chunk metadata")

    chunk_start = float(sample["chunk_start"]) / sample_rate
    chunk_end = float(sample["chunk_end"]) / sample_rate
    if chunk_end < chunk_start:
        raise ValueError("chunk_end must not precede chunk_start")

    lookup = alignment_reader.get_words(utterance_id)
    selected = []
    for word in lookup.words:
        if word.start >= chunk_start - tolerance and word.end <= chunk_end + tolerance:
            phone_ids = word_to_phone_ids.get(word.text)
            if phone_ids is None:
                raise ChunkKeywordSelectionError(
                    sample["key"],
                    speaker_slot,
                    utterance_id,
                    f"missing_lexicon:{word.text}",
                )
            selected.append(list(phone_ids))
    if not selected:
        raise ChunkKeywordSelectionError(
            sample["key"],
            speaker_slot,
            utterance_id,
            "no_complete_aligned_word",
        )
    return selected, utterance_id


def _log_chunk_stats(stats, sample, final=False):
    if not stats:
        return
    suffix = " final" if final else ""
    logging.info(
        "textual chunk selection%s rank=%s worker=%s stats=%s",
        suffix,
        sample.get("rank", 0),
        sample.get("worker_id", 0),
        dict(stats),
    )


def sample_textual_cue(
    data,
    resource_path,
    key_field,
    cue_format,
    format_args=None,
    scope="speaker",
    required=True,
):
    """Attach `textual_spk{i}` arrays from a configured textual resource.

    With ``selection.chunk_aware`` enabled by the chunked dataset
    configuration, random training keywords are selected only from complete
    word alignments in the shared audio crop. An
    explicitly configured ``discard`` policy rejects the entire mixture if any
    target source has no usable aligned word.
    """
    if scope != "speaker":
        raise ValueError("textual cue currently supports scope='speaker'")
    if cue_format != ("json", "dae_keyword_phoneme"):
        raise ValueError(f"Unsupported textual cue format: {cue_format}")

    format_args = format_args or {}
    selection = format_args.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("DAE keyword textual cue requires a 'selection' mapping")
    chunk_options = _chunk_selection_options(selection)

    if resource_path not in _TEXTUAL_RESOURCE_CACHE:
        _TEXTUAL_RESOURCE_CACHE[resource_path] = load_json(resource_path)
    cue_resource = _TEXTUAL_RESOURCE_CACHE[resource_path]

    alignment_reader = None
    word_to_phone_ids = None
    stats = {}
    last_sample = None
    if chunk_options is not None:
        alignment_reader = LibriSpeechAlignmentReader(
            chunk_options["alignment_path"])
        word_to_phone_ids = _get_word_to_phone_ids(
            chunk_options["word2lexicon_path"])

    try:
        for sample in data:
            last_sample = sample
            speaker_slots = _speaker_slots(sample)
            if not speaker_slots:
                if required:
                    raise KeyError("sample has no speaker slots (spk1, spk2, ...)")
                yield sample
                continue

            if chunk_options is None:
                for speaker_slot in speaker_slots:
                    lookup_key = _lookup_key(sample, speaker_slot, key_field)
                    if lookup_key not in cue_resource:
                        if required:
                            raise KeyError(
                                f"textual cue not found: {lookup_key}")
                        continue
                    try:
                        sample[f"textual_{speaker_slot}"] = _read_dae_keyword_phoneme(
                            cue_resource[lookup_key], selection)
                    except Exception as error:
                        # Preserve the whole-utterance optional-cue behavior:
                        # malformed entries are local when the cue is optional.
                        logging.warning("Failed to read textual cue %s: %s",
                                        lookup_key, error)
                        if required:
                            raise
                yield sample
                continue

            stats["mixture_draws"] = stats.get("mixture_draws", 0) + 1
            failure = None
            selected_cues = {}
            for speaker_slot in speaker_slots:
                stats["target_speaker_draws"] = (
                    stats.get("target_speaker_draws", 0) + 1)
                lookup_key = _lookup_key(sample, speaker_slot, key_field)
                if lookup_key not in cue_resource:
                    stats["missing_textual_resource"] = (
                        stats.get("missing_textual_resource", 0) + 1)
                    failure = ChunkKeywordSelectionError(
                        sample["key"], speaker_slot,
                        _utterance_id_for_slot(sample, speaker_slot),
                        "missing_textual_resource",
                    )
                    break
                try:
                    labels, utterance_id = _complete_chunk_word_labels(
                        sample,
                        speaker_slot,
                        alignment_reader,
                        word_to_phone_ids,
                        chunk_options["tolerance"],
                    )
                    stats["aligned_target_speakers"] = (
                        stats.get("aligned_target_speakers", 0) + 1)
                    selected_cues[speaker_slot] = np.asarray(
                        [phone_id for word in _select_random_word_labels(
                            labels, selection.get("word_range", [2, 6]))
                         for phone_id in word],
                        dtype=np.int64,
                    )
                except AlignmentNotFoundError as error:
                    utterance_id = _utterance_id_for_slot(sample, speaker_slot)
                    stats["missing_alignment"] = (
                        stats.get("missing_alignment", 0) + 1)
                    failure = ChunkKeywordSelectionError(
                        sample["key"], speaker_slot, utterance_id,
                        "missing_alignment",
                    )
                    break
                except ChunkKeywordSelectionError as error:
                    stats[error.reason.split(":", 1)[0]] = (
                        stats.get(error.reason.split(":", 1)[0], 0) + 1)
                    failure = error
                    break

            if failure is not None:
                stats["discarded_mixtures"] = (
                    stats.get("discarded_mixtures", 0) + 1)
                if chunk_options["no_word_policy"] == "discard":
                    if stats["mixture_draws"] % 1000 == 0:
                        _log_chunk_stats(stats, sample)
                    continue
                raise failure

            for speaker_slot, cue in selected_cues.items():
                sample[f"textual_{speaker_slot}"] = cue
            if stats["mixture_draws"] % 1000 == 0:
                _log_chunk_stats(stats, sample)
            yield sample
    finally:
        if alignment_reader is not None:
            alignment_reader.close()
        if chunk_options is not None and last_sample is not None:
            _log_chunk_stats(stats, last_sample, final=True)
