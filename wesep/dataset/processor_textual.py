# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

import logging
import random
import re

import numpy as np

from wesep.utils.file_utils import load_json

# Textual resources are loaded once in each DataLoader worker process.
_TEXTUAL_RESOURCE_CACHE = {}


def _read_dae_keyword_phoneme(item, selection):
    """Select DAE keyword words and return one flattened phoneme sequence."""
    labels = item.get("phn_label") if isinstance(item, dict) else None
    if not isinstance(labels, list) or not labels:
        raise ValueError("DAE keyword cue requires non-empty 'phn_label'")
    if any(not isinstance(word, list) or not word for word in labels):
        raise ValueError("Each DAE keyword word requires phoneme IDs")

    selection_type = selection.get("type")
    if selection_type == "random":
        word_range = selection.get("word_range", [2, 6])
        if (not isinstance(word_range, list) or len(word_range) != 2
                or not all(isinstance(value, int) for value in word_range)
                or word_range[0] <= 0 or word_range[1] < word_range[0]):
            raise ValueError(
                "DAE random selection requires a valid 'word_range'")
        word_count = min(random.randint(*word_range), len(labels))
        start = random.randint(0, len(labels) - word_count)
        selected = labels[start:start + word_count]
    elif selection_type == "fixed":
        candidates = item.get("kw_candidate")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(
                "DAE fixed selection requires non-empty 'kw_candidate'")
        if any(not isinstance(index, int) or index < 0 or index >= len(labels)
               for index in candidates):
            raise IndexError("DAE keyword candidate is outside 'phn_label'")
        selected = [labels[index] for index in candidates]
    else:
        raise ValueError(
            f"Unsupported DAE keyword selection: {selection_type}")

    phonemes = [value for word in selected for value in word]
    if not phonemes:
        raise ValueError("DAE keyword selection produced no phoneme IDs")
    return np.asarray(phonemes, dtype=np.int64)


def sample_textual_cue(
    data,
    resource_path,
    key_field,
    cue_format,
    format_args=None,
    scope="speaker",
    required=True,
):
    """Attach `textual_spk{i}` tensors from a configured textual resource.

    Args:
        data: samples containing ordered `spk{i}` fields.
        resource_path: JSON mapping speaker lookup keys to textual records.
        key_field: `spk_id` or `mix_spk_id`.
        cue_format: currently (`json`, `dae_keyword_phoneme`).
        format_args: optional arguments owned by the selected representation.

    Returns:
        Samples with available 1D `textual_spk{i}` arrays attached.
    """
    if scope != "speaker":
        raise ValueError("textual cue currently supports scope='speaker'")

    # Select the representation reader and validate only its own arguments.
    format_args = format_args or {}
    if cue_format == ("json", "dae_keyword_phoneme"):
        item_reader = _read_dae_keyword_phoneme
        selection = format_args.get("selection")
        if not isinstance(selection, dict):
            raise ValueError(
                "DAE keyword textual cue requires a 'selection' mapping.")
        reader_args = (selection, )
    else:
        raise ValueError(f"Unsupported textual cue format: {cue_format}")
    if resource_path not in _TEXTUAL_RESOURCE_CACHE:
        _TEXTUAL_RESOURCE_CACHE[resource_path] = load_json(resource_path)
    cue_resource = _TEXTUAL_RESOURCE_CACHE[resource_path]

    for sample in data:
        # Resolve the ordered target speaker slots in the current sample.
        spk_slots = [key for key in sample if re.fullmatch(r"spk\d+", key)]
        spk_slots.sort(key=lambda key: int(key[3:]))
        if not spk_slots:
            if required:
                raise KeyError("sample has no speaker slots (spk1, spk2, ...)")
            yield sample
            continue

        for slot in spk_slots:
            # Build the same dataset-level lookup key used by other cue types.
            if key_field == "spk_id":
                lookup_key = sample[slot]
            elif key_field == "mix_spk_id":
                mix_key = sample.get(f"source_key_{slot}",
                                     sample.get("key", None))
                if mix_key is None:
                    raise KeyError("sample missing 'key' for mix_spk_id cue")
                lookup_key = f"{mix_key}::{sample[slot]}"
            else:
                raise ValueError(
                    f"Unsupported key_field for textual cue: {key_field}")

            if lookup_key not in cue_resource:
                if required:
                    raise KeyError(f"textual cue not found: {lookup_key}")
                continue

            # Decode the representation and keep optional-cue failures local.
            try:
                textual = item_reader(cue_resource[lookup_key], *reader_args)
            except Exception as error:
                logging.warning("Failed to read textual cue %s: %s",
                                lookup_key, error)
                if required:
                    raise
                continue
            sample[f"textual_{slot}"] = textual

        yield sample
