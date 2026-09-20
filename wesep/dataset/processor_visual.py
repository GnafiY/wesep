# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

import logging
import re

import numpy as np
import torch
from torchvision.io import read_video

from wesep.utils.file_utils import load_json


def _audio_seconds(sample, spk_slot):
    ratio = sample.get(f"chunk_ratio_{spk_slot}",
                       sample.get("chunk_ratio", None))
    if ratio is not None:
        return ratio["orig_len"] / sample["sample_rate"]
    return sample["wav_mix"].shape[-1] / sample["sample_rate"]


def _align_last_dim_to_audio_duration(x, duration_sec, sample, spk_slot):
    """Align a precomputed temporal cue to the full audio duration."""
    if x.shape[-1] <= 0:
        raise ValueError("temporal cue must have non-zero last dimension")
    if duration_sec is None:
        return x

    duration_sec = float(duration_sec)
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}")

    cur_len = x.shape[-1]
    target_len = int(
        round(_audio_seconds(sample, spk_slot) * cur_len / duration_sec))
    if target_len <= 0:
        raise ValueError(
            f"target temporal length must be positive, got {target_len}")
    if cur_len == target_len:
        return x
    if cur_len > target_len:
        return x[..., :target_len]

    pad_len = target_len - cur_len
    pad = x[..., -1:].expand(*x.shape[:-1], pad_len)
    return torch.cat([x, pad], dim=-1)


def _read_raw_video_item(item, sample, spk_slot):
    """Read one video cue as [H, W, C, T] for the selected speaker slot."""
    video_path = item["path"]
    video, _, info = read_video(video_path, pts_unit="sec")
    fps = info["video_fps"]

    # video: [T, H, W, C] uint8
    if video.numel() == 0:
        raise RuntimeError(f"Empty video: {video_path}")

    # First match full-utterance duration, then apply chunk-level crop.
    audio_sec = _audio_seconds(sample, spk_slot)
    video_sec = video.shape[0] / fps
    if video_sec < audio_sec:
        need_sec = audio_sec - video_sec
        need_frames = int(round(need_sec * fps))
        last_frame = video[-1:].repeat(need_frames, 1, 1, 1)
        video = torch.cat([video, last_frame], dim=0)
    elif video_sec > audio_sec:
        max_frames = int(round(audio_sec * fps))
        video = video[:max_frames]

    ratio = sample.get(f"chunk_ratio_{spk_slot}",
                       sample.get("chunk_ratio", None))
    if ratio is not None:
        start_ratio = ratio["start_ratio"]
        end_ratio = ratio["end_ratio"]

        num_frames = video.shape[0]
        start_f = int(start_ratio * num_frames)
        end_f = int(end_ratio * num_frames)

        start_f = max(0, min(start_f, num_frames))
        crop_end = max(start_f + 1, min(end_f, num_frames))
        video = video[start_f:crop_end]
        if end_f > num_frames:
            last_frame = video[-1:].repeat(end_f - num_frames, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)

    # Keep decoded frames compact until the visual frontend runs on device.
    return video.permute(1, 2, 3, 0)  # [H, W, C, T] uint8


def _read_muse_frontend_item(item, sample, spk_slot):
    """Read one precomputed visual feature and align its final time axis."""
    feat = np.load(item["path"])
    feat = torch.from_numpy(np.asarray(feat)).float()
    if feat.ndim < 2:
        raise ValueError(
            f"muse_frontend visual cue must have at least 2 dims, got {tuple(feat.shape)}"
        )
    feat = _align_last_dim_to_audio_duration(
        feat,
        item.get("duration_sec", None),
        sample,
        spk_slot,
    )
    ratio = sample.get(f"chunk_ratio_{spk_slot}",
                       sample.get("chunk_ratio", None))
    if ratio is not None:
        num_frames = feat.shape[-1]
        start_f = int(ratio["start_ratio"] * num_frames)
        end_f = int(ratio["end_ratio"] * num_frames)
        start_f = max(0, min(start_f, num_frames))
        crop_end = max(start_f + 1, min(end_f, num_frames))
        feat = feat[..., start_f:crop_end]
        if end_f > num_frames:
            pad = feat[..., -1:].expand(*feat.shape[:-1], end_f - num_frames)
            feat = torch.cat([feat, pad], dim=-1)
    return feat


# module-level cache (per worker process)
_SPK_RESOURCE_CACHE = {}


def sample_visual_cue(
    data,
    resource_path,
    key_field,
    cue_format,
    scope="speaker",
    required=True,
):
    """Attach `visual_spk{i}` tensors from MP4 or precomputed npy items.

    Args:
        data: samples containing audio timing metadata and `spk{i}` fields.
        resource_path: JSON mapping lookup keys to visual item lists.
        key_field: `spk_id` or `mix_spk_id`.
        cue_format: (`mp4`, `raw_video`) or (`npy`, `muse_frontend`).

    Returns:
        Samples with available `visual_spk{i}` tensors attached.
    """
    if scope != "speaker":
        raise ValueError("visual cue currently supports scope='speaker'")

    # Select the format decoder and load its index once per worker.
    if cue_format == ("mp4", "raw_video"):
        item_reader = _read_raw_video_item
    elif cue_format == ("npy", "muse_frontend"):
        item_reader = _read_muse_frontend_item
    else:
        raise ValueError(f"Unsupported visual cue format: {cue_format}")
    if resource_path not in _SPK_RESOURCE_CACHE:
        _SPK_RESOURCE_CACHE[resource_path] = load_json(resource_path)
    spk_resource = _SPK_RESOURCE_CACHE[resource_path]

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
                    f"Unsupported key_field for visual cue: {key_field}")

            if lookup_key not in spk_resource:
                if required:
                    raise KeyError(f"fixed visual cue not found: {lookup_key}")
                continue

            items = spk_resource[lookup_key]
            if not items:
                if required:
                    raise RuntimeError(f"empty fixed visual cue: {lookup_key}")
                continue

            # Decode and time-align the first fixed visual item.
            try:
                visual = item_reader(items[0], sample, slot)
            except Exception as e:
                logging.warning(
                    f"Failed to read visual cue: {items[0]}, err={e}")
                if required:
                    raise
                continue

            sample[f"visual_{slot}"] = visual
        yield sample
