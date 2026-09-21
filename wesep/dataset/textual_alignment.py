# Copyright (c) 2026 Haoyu Li (haoyu.li.cs@sjtu.edu.cn)
#
# SPDX-License-Identifier: Apache-2.0

"""Word-level LibriSpeech alignment access for textual TSE data pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile


@dataclass(frozen=True)
class AlignedWord:
    """One lower-cased word with a half-open time interval in seconds."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class AlignmentLookup:
    """Alignment words and which resource supplied them."""

    words: tuple[AlignedWord, ...]
    source: str
    primary_available: bool
    fallback_available: bool


class AlignmentNotFoundError(KeyError):
    """Raised when an utterance is absent from both alignment resources."""

    def __init__(self, utterance_id, primary_available, fallback_available):
        self.utterance_id = utterance_id
        self.primary_available = primary_available
        self.fallback_available = fallback_available
        super().__init__(
            f"No alignment for {utterance_id} "
            f"(primary_available={primary_available}, "
            f"fallback_available={fallback_available})")


class _AlignmentSource:
    """Read chapter records from one zip archive or extracted directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Alignment resource is absent: {self.path}")

        self._archive = None
        self._members = {}
        self._directory_members = {}
        self._chapter_cache = {}
        if self.path.is_dir():
            self._directory_members = {
                member.name.removesuffix(".alignment.txt"): member
                for member in self.path.rglob("*.alignment.txt")
            }
        else:
            self._archive = zipfile.ZipFile(self.path)
            self._members = {
                Path(member).name.removesuffix(".alignment.txt"): member
                for member in self._archive.namelist()
                if member.endswith(".alignment.txt")
            }

    @staticmethod
    def _parse_records(raw_text: str) -> dict[str, tuple[AlignedWord, ...]]:
        records = {}
        for raw_line in raw_text.splitlines():
            fields = raw_line.replace('"', "").split()
            if len(fields) != 3:
                raise ValueError(f"Malformed alignment record: {raw_line}")
            utterance_id, raw_words, raw_ends = fields
            # Official MFA records timestamp every comma position, including
            # empty non-word intervals. Corrected Qwen records retain empty
            # leading/trailing comma fields but timestamp only spoken words.
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

    def _load_chapter(self, chapter_id: str) -> dict[str, tuple[AlignedWord,
                                                                ...]]:
        if chapter_id in self._chapter_cache:
            return self._chapter_cache[chapter_id]

        raw_text = None
        if self._archive is None:
            member = self._directory_members.get(chapter_id)
            if member is not None:
                raw_text = member.read_text(encoding="utf-8")
        else:
            member = self._members.get(chapter_id)
            if member is not None:
                raw_text = self._archive.read(member).decode("utf-8")

        records = self._parse_records(raw_text) if raw_text is not None else {}
        self._chapter_cache[chapter_id] = records
        return records

    def get(self, utterance_id: str) -> tuple[AlignedWord, ...] | None:
        chapter_id = "-".join(utterance_id.split("-")[:2])
        return self._load_chapter(chapter_id).get(utterance_id)

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()


class LibriSpeechAlignmentReader:
    """Look up word alignments from primary and optional fallback resources.

    Both resources may be a zip archive or an extracted LibriSpeech alignment
    directory. Archives are read in place and are never unpacked by workers.
    """

    def __init__(
        self,
        primary_path: str | Path,
        fallback_path: str | Path | None = None,
    ):
        self._primary = _AlignmentSource(primary_path)
        self._fallback = (_AlignmentSource(fallback_path)
                          if fallback_path is not None else None)

    def get_words(self, utterance_id: str) -> AlignmentLookup:
        primary_words = self._primary.get(utterance_id)
        primary_available = primary_words is not None
        if primary_available:
            return AlignmentLookup(
                words=primary_words,
                source="primary",
                primary_available=True,
                fallback_available=False,
            )

        fallback_words = (self._fallback.get(utterance_id)
                          if self._fallback is not None else None)
        fallback_available = fallback_words is not None
        if fallback_available:
            return AlignmentLookup(
                words=fallback_words,
                source="fallback",
                primary_available=False,
                fallback_available=True,
            )

        raise AlignmentNotFoundError(
            utterance_id,
            primary_available=False,
            fallback_available=False,
        )

    def close(self) -> None:
        self._primary.close()
        if self._fallback is not None:
            self._fallback.close()

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
