# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

import logging
import re

import numpy as np
import torch

from wesep.utils.file_utils import load_json

# module-level cache (per worker process)
_CUE_RESOURCE_CACHE = {}


def _load_named_npy(path):
    """Load a spatial npy/npz payload containing named fields."""
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        payload = {key: data[key] for key in data.files}
        data.close()
        return payload
    if data.shape == ():
        data = data.item()
    if not hasattr(data, "__contains__"):
        raise ValueError(f"spatial npy must contain named fields: {path}")
    return data


def _read_spatial_item(item, fields, field_defaults, required, sample,
                       spk_slot):
    """Read one spatial item and return [F] or time-aligned [F, T]."""
    payload = _load_named_npy(item["path"])
    duration_sec = item.get("duration_sec", None)
    ratio = sample.get(f"chunk_ratio_{spk_slot}",
                       sample.get("chunk_ratio", None))
    values = []

    # Read only the declared spatial fields in their given order.
    for field in fields:
        if field in payload:
            value = np.asarray(payload[field], dtype=np.float32)
        elif field in field_defaults:
            value = np.asarray(field_defaults[field], dtype=np.float32)
        elif required:
            raise KeyError(f"spatial field missing: {field}")
        else:
            return None

        if value.ndim == 1 and value.shape[0] <= 0:
            raise ValueError("temporal cue field must have non-zero length")

        # Match temporal fields to the original audio duration.
        if value.ndim == 1 and duration_sec is not None:
            duration = float(duration_sec)
            if duration <= 0:
                raise ValueError(
                    f"duration_sec must be positive, got {duration}")
            cur_len = value.shape[0]
            if ratio is not None:
                audio_len = ratio["orig_len"]
            else:
                audio_len = sample["wav_mix"].shape[-1]
            audio_sec = audio_len / sample["sample_rate"]
            target_len = int(round(audio_sec * cur_len / duration))
            if target_len <= 0:
                raise ValueError(
                    f"target temporal length must be positive, got {target_len}"
                )
            if cur_len > target_len:
                value = value[:target_len]
            elif cur_len < target_len:
                pad = np.full(target_len - cur_len,
                              value[-1],
                              dtype=value.dtype)
                value = np.concatenate([value, pad], axis=0)

        # Apply the audio chunk and right-padding range.
        if value.ndim == 1 and ratio is not None:
            seq_len = value.shape[0]
            start = int(ratio["start_ratio"] * seq_len)
            end = int(ratio["end_ratio"] * seq_len)
            start = max(0, min(start, seq_len))
            crop_end = max(start + 1, min(end, seq_len))
            value = value[start:crop_end]
            if end > seq_len:
                pad = np.full(end - seq_len, value[-1], dtype=value.dtype)
                value = np.concatenate([value, pad], axis=0)
        values.append(value)

    # Stack scalar fields as [F] or temporal fields as [F, T].
    seq_lens = [value.shape[0] for value in values if value.ndim == 1]
    if any(value.ndim > 1 for value in values):
        shapes = [tuple(value.shape) for value in values]
        raise ValueError(f"spatial fields must be scalar or 1D, got {shapes}")
    if not seq_lens:
        return torch.tensor([float(value) for value in values],
                            dtype=torch.float32)
    if len(set(seq_lens)) != 1:
        raise ValueError(
            f"spatial sequence fields have different lengths: {seq_lens}")

    seq_len = seq_lens[0]
    rows = []
    for value in values:
        if value.ndim == 0:
            rows.append(np.full(seq_len, float(value), dtype=np.float32))
        else:
            rows.append(value)
    return torch.from_numpy(np.stack(rows, axis=0)).float()


def sample_fixed_spatial_cue(
    data,
    resource_path,
    fields,
    key_field,
    field_defaults=None,
    scope="speaker",
    required=True,
):
    """Read configured spatial fields and attach `spatial_spk{i}` tensors.

    Args:
        data: samples containing audio timing metadata and `spk{i}` fields.
        resource_path: JSON mapping lookup keys to spatial npy/npz items.
        fields: ordered scalar or 1D fields to read and stack.
            Angular fields must be stored in radians.
        key_field: `spk_id` or `mix_spk_id`.

    Returns:
        Samples with `[F]` static or `[F, T]` temporal spatial tensors.
    """
    if scope != "speaker":
        raise ValueError("spatial cue currently supports scope='speaker'")

    # Load the spatial index once in each DataLoader worker.
    if resource_path not in _CUE_RESOURCE_CACHE:
        _CUE_RESOURCE_CACHE[resource_path] = load_json(resource_path)
    cue_resource = _CUE_RESOURCE_CACHE[resource_path]
    field_defaults = field_defaults or {}

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
            # Build the offline or original online utterance lookup key.
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
                    f"Unsupported key_field for spatial cue: {key_field}")

            if lookup_key not in cue_resource:
                if required:
                    raise KeyError(f"spatial cue not found: {lookup_key}")
                continue

            item = cue_resource[lookup_key]
            if not item:
                if required:
                    raise RuntimeError(f"empty spatial cue: {lookup_key}")
                continue

            # Read, time-align, and stack the configured spatial fields.
            try:
                spatial = _read_spatial_item(
                    item,
                    fields,
                    field_defaults,
                    required,
                    sample,
                    slot,
                )
            except Exception as e:
                logging.warning(f"Failed to read spatial cue: {item}, err={e}")
                if required:
                    raise
                continue

            if spatial is not None:
                sample[f"spatial_{slot}"] = spatial

        yield sample
