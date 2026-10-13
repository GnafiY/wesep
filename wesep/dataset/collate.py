# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import torch

BASE_COLLECT_KEYS = {
    # ===== Metadata =====
    "spk": {
        "key_tpl": "spk{}",
        "axis": "spk",
        "required": True,
        "value_type": "meta",
    },
    "key": {
        "axis": "mix",
        "required": True,
        "value_type": "meta",
    },
    "num_speaker": {
        "axis": "mix",
        "required": True,
        "value_type": "meta",
    },
    "speaker_label": {
        "key_tpl": "spk{}_label",
        "axis": "spk",
        "required": False,
        "value_type": "tensor",
        "default_shape": (),
        "fill_value": -100,
    },

    # ===== Model features =====
    # Waveforms always pad to the longest item in the batch. Chunk training is
    # usually equal-length already; whole-utterance training relies on this.
    "wav_mix": {
        "axis": "mix",
        "required": True,
        "align": "max",
        "value_type": "tensor",
        "pad_mode": "constant",
        "fill_value": 0.0,
    },
    "wav_target": {
        "key_tpl": "wav_spk{}",
        "axis": "spk",
        "required": True,
        "align": "max",
        "value_type": "tensor",
        "pad_mode": "constant",
        "fill_value": 0.0,
    },

    # ===== Audio cue =====
    "audio_aux": {
        "key_tpl": "audio_spk{}",
        "axis": "spk",
        "required": False,
        "align": "min",
        "value_type": "tensor",
        "pad_mode": "constant",
        "default_shape": None,
        "fill_value": 0.0,  # All-missing shape comes from cues.yaml.
        "emit_present": True,
    },

    # ===== Spatial cue =====
    "spatial_aux": {
        "key_tpl": "spatial_spk{}",
        "axis": "spk",
        "required": False,
        "align": "max",
        "value_type": "tensor",
        "pad_mode": "edge",
        "default_shape": None,
        "fill_value": -999.0,  # All-missing shape comes from cues.yaml.
        "emit_present": True,
    },

    # ===== Visual cue =====
    "visual_aux": {
        "key_tpl": "visual_spk{}",
        "axis": "spk",
        "required": False,
        "align": "max",
        "value_type": "tensor",
        "pad_mode": "edge",
        "default_shape": None,
        "fill_value": 0.0,  # All-missing shape comes from cues.yaml.
        "emit_present": True,
    },

    # ===== Textual cue =====
    "textual_aux": {
        "key_tpl": "textual_spk{}",
        "axis": "spk",
        "required": False,
        "align": "max",
        "value_type": "tensor",
        "pad_mode": "constant",
        "default_shape": None,
        "fill_value": 0,  # All-missing shape comes from cues.yaml.
        "emit_present": True,
    },
}

BSS_COLLECT_KEYS = {
    key: dict(BASE_COLLECT_KEYS[key])
    for key in ("wav_mix", "spk", "key", "num_speaker")
}
BSS_COLLECT_KEYS["wav_sources"] = dict(BASE_COLLECT_KEYS["wav_target"])

AUX_KEY_MAP = {
    "audio": "audio_aux",
    "spatial": "spatial_aux",
    "visual": "visual_aux",
    "textual": "textual_aux",
}


def build_collect_keys(cues_conf, train_conf, base_table):
    """
    Args:
        cues_conf: dict loaded from cues.yaml["cues"]
        train_conf: model conf (expects dataset_args.cues)
        base_table: BASE_COLLECT_KEYS

    Returns:
        collect_keys: dict for tse_collate_fn
    """
    collect_keys = {}
    cues_conf = cues_conf or {}
    if "cues" in cues_conf:
        cues_conf = cues_conf["cues"]

    # ---- 1) Fixed keys are always collected ----
    for k in [
            "wav_mix", "wav_target", "spk", "key", "num_speaker",
            "speaker_label"
    ]:
        if k not in base_table:
            raise KeyError(f"[collect_keys] base_table missing fixed key: {k}")
        collect_keys[k] = dict(base_table[k])

    # ---- 2) Cue keys are requested by training config ----
    want_cues = train_conf.get("cues", {})
    for cue_name, want in want_cues.items():
        if not want.get("use", False):
            continue

        cue_cfg = cues_conf.get(cue_name)
        if cue_cfg is None:
            raise RuntimeError(
                f"[collect_keys] Training requires cue {cue_name!r}, "
                f"but dataset cues.yaml does not provide it.")

        if cue_cfg.get("scope", "speaker") != "speaker":
            continue

        if cue_name not in AUX_KEY_MAP:
            raise RuntimeError(
                f"[collect_keys] Unknown cue modality {cue_name!r}. "
                f"Known: {list(AUX_KEY_MAP.keys())}")

        aux_key = AUX_KEY_MAP[cue_name]
        if aux_key not in base_table:
            raise KeyError(
                f"[collect_keys] base_table missing aux key spec: {aux_key}")

        spec = dict(base_table[aux_key])
        required = want.get("required", spec.get("required", True))
        guaranteed = cue_cfg.get("guaranteed", True)
        if required and not guaranteed:
            raise RuntimeError(
                f"[collect_keys] Training requires cue {cue_name!r} to be present "
                f"in every sample, but dataset cues.yaml marks it as optional."
            )

        # Cue defaults are data-side fallbacks; train config may override them.
        cue_collate = cue_cfg.get("collate", {})
        for k in ("default_shape", "fill_value", "emit_present"):
            if k in cue_collate:
                spec[k] = cue_collate[k]
            if k in want:
                spec[k] = want[k]
        spec["required"] = required

        collect_keys[aux_key] = spec

    for key, overrides in train_conf.get("collate", {}).items():
        if key not in collect_keys:
            raise KeyError(
                f"[collect_keys] Cannot override unregistered key: {key}")
        collect_keys[key].update(overrides)

    return collect_keys


def _to_tensor(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise TypeError(f"[collate] Unsupported type: {type(x)}")


def _pad_or_crop_to_len(x, target_len, fill_value=0.0, pad_mode="constant"):
    cur_len = x.shape[-1]
    if cur_len == target_len:
        return x
    if cur_len > target_len:
        return x[..., :target_len]

    pad_len = target_len - cur_len
    if pad_mode == "edge" and cur_len > 0:
        pad = x[..., -1:].expand(*x.shape[:-1], pad_len)
    elif pad_mode == "constant":
        pad_shape = list(x.shape)
        pad_shape[-1] = pad_len
        pad = torch.full(
            tuple(pad_shape),
            fill_value,
            dtype=x.dtype,
            device=x.device,
        )
    else:
        raise ValueError(f"[collate] Unknown pad_mode: {pad_mode}")
    return torch.cat([x, pad], dim=-1)


def _fallback_tensor(ref_tensor, default_shape, fill_value, out_key):
    if ref_tensor is not None:
        return torch.full(
            tuple(ref_tensor.shape),
            fill_value,
            dtype=ref_tensor.dtype,
            device=ref_tensor.device,
        )
    if default_shape is None:
        raise RuntimeError(
            f"[collate] Cannot infer fallback shape for {out_key!r}: "
            "no real sample exists in batch and no default_shape is set.")
    return torch.full(tuple(default_shape), fill_value)


def bss_collate_fn(batch, collect_keys):
    """Collect mix/spk fields without expanding the mixture batch axis.

    Like tse_collate_fn, collect_keys defines each output field. Speaker
    tensors are grouped per mixture as [S, T], giving [B, S, T] after stacking.
    """
    if not batch:
        return {}

    # ---- 1) Keep source counts and valid lengths at mixture level ----
    num_speakers = [int(sample["num_speaker"]) for sample in batch]
    if len(set(num_speakers)) != 1:
        raise ValueError("BSS requires the same source count within a batch")
    valid_lengths = []
    for sample in batch:
        mix = _to_tensor(sample["wav_mix"])
        if mix.ndim != 2 or mix.shape[0] != 1:
            raise ValueError("BSS requires mono [1, T] mixtures")
        valid_length = int(sample.get("valid_length", mix.shape[-1]))
        if not 0 < valid_length <= mix.shape[-1]:
            raise ValueError(f"Invalid valid_length={valid_length}")
        valid_lengths.append(valid_length)

    new_batch = {"valid_lengths": torch.tensor(valid_lengths, dtype=torch.long)}

    # ---- 2) Collect registered fields with the same specs as TSE ----
    for out_key, spec in collect_keys.items():
        key_tpl = spec.get("key_tpl", None)
        axis = spec.get("axis", "mix")
        align = spec.get("align", None)
        value_type = spec["value_type"]
        fill_value = spec.get("fill_value", 0.0)
        pad_mode = spec.get("pad_mode", "constant")
        batch_vals = []
        for sample, ns in zip(batch, num_speakers):
            if axis == "mix":
                values = [sample[out_key]]
            elif axis == "spk":
                values = [sample[key_tpl.format(i)] for i in range(1, ns + 1)]
            else:
                raise ValueError(f"[collate] Unknown axis: {axis}")

            if value_type == "meta":
                batch_vals.append(values[0] if axis == "mix" else values)
            elif value_type == "tensor":
                values = [_to_tensor(value) for value in values]
                if out_key == "wav_sources" and any(
                        value.shape != sample["wav_mix"].shape for value in values):
                    raise ValueError("BSS sources must match mixture shape")
                batch_vals.append(values[0] if axis == "mix"
                                  else torch.cat(values, dim=0))
            else:
                raise ValueError(f"[collate] Unknown value_type: {value_type}")

        if value_type == "tensor":
            if align is not None:
                lengths = [value.shape[-1] for value in batch_vals]
                if align == "max":
                    target_len = max(lengths)
                elif align == "min":
                    target_len = min(lengths)
                else:
                    raise ValueError(f"[collate] Unknown align mode: {align}")
                batch_vals = [
                    _pad_or_crop_to_len(value, target_len, fill_value, pad_mode)
                    for value in batch_vals
                ]
            new_batch[out_key] = torch.stack(batch_vals, dim=0)
        else:
            new_batch[out_key] = batch_vals
    return new_batch


def tse_collate_fn(batch, collect_keys):
    """
    Speaker-axis collate:
      - each mix generates num_speaker batch samples
      - mix-axis features are copied
      - speaker-axis features are indexed
      - metadata is also copied

    Args:
        batch: list[dict]
        collect_keys: dict

    Returns:
        new_batch: dict
    """
    if len(batch) == 0:
        return {}

    # ---- 1) Determine speaker expansion for each mixture ----
    num_speakers = []
    for s in batch:
        if "num_speaker" not in s:
            raise RuntimeError(
                f"[collate] Missing required key 'num_speaker' in sample: {s.get('key')}"
            )
        num_speakers.append(int(s["num_speaker"]))

    new_batch = {}

    # Mixture-level valid lengths expand once per target speaker. Whole-utterance
    # samples have no chunk metadata and therefore use their full waveform.
    valid_lengths = []
    for sample, num_speaker in zip(batch, num_speakers):
        valid_length = int(sample.get("valid_length", sample["wav_mix"].shape[-1]))
        if valid_length <= 0 or valid_length > sample["wav_mix"].shape[-1]:
            raise ValueError(
                f"[collate] Invalid valid_length={valid_length} for "
                f"sample {sample.get('key')} with waveform length "
                f"{sample['wav_mix'].shape[-1]}")
        valid_lengths.extend([valid_length] * num_speaker)
    new_batch["valid_lengths"] = torch.as_tensor(valid_lengths, dtype=torch.long)

    # ---- 2) Collect each registered output field ----
    for out_key, spec in collect_keys.items():
        key_tpl = spec.get("key_tpl", None)
        axis = spec.get("axis", "mix")
        align = spec.get("align", None)
        required = spec.get("required", True)
        value_type = spec["value_type"]
        default_shape = spec.get("default_shape", None)
        fill_value = spec.get("fill_value", 0.0)
        pad_mode = spec.get("pad_mode", "constant")
        emit_present = spec.get("emit_present", False)

        flat_vals = []
        flat_lens = []
        present = []
        ref_tensor = None

        # Metadata stays as a Python list after speaker expansion.
        if value_type == "meta":
            for bidx, s in enumerate(batch):
                ns = num_speakers[bidx]
                if axis == "mix":
                    if out_key not in s:
                        if required:
                            raise RuntimeError(
                                f"[collate] Missing required key {out_key!r} in sample: {s.get('key')}"
                            )
                        v = None
                    else:
                        v = s[out_key]
                    flat_vals.extend([v] * ns)
                elif axis == "spk":
                    for i in range(1, ns + 1):
                        k = key_tpl.format(i)
                        if k not in s:
                            if required:
                                raise RuntimeError(
                                    f"[collate] Missing required key {k!r} in sample: {s.get('key')}"
                                )
                            v = None
                        else:
                            v = s[k]
                        flat_vals.append(v)
                else:
                    raise ValueError(f"[collate] Unknown axis: {axis}")
            new_batch[out_key] = flat_vals
            continue

        if value_type != "tensor":
            raise ValueError(f"[collate] Unknown value_type: {value_type}")

        # Tensor fields are flattened first, then aligned and stacked.
        for bidx, s in enumerate(batch):
            ns = num_speakers[bidx]
            if axis == "mix":
                if out_key not in s:
                    if required:
                        raise RuntimeError(
                            f"[collate] Missing required key {out_key!r} in sample: {s.get('key')}"
                        )
                    x = None
                else:
                    x = _to_tensor(s[out_key])

                for _ in range(ns):
                    flat_vals.append(x)
                    present.append(x is not None)
                    if x is not None:
                        if align is not None:
                            flat_lens.append(x.shape[-1])
                        if ref_tensor is None:
                            ref_tensor = x

            elif axis == "spk":
                for i in range(1, ns + 1):
                    k = key_tpl.format(i)
                    if k not in s:
                        if required:
                            raise RuntimeError(
                                f"[collate] Missing required key {k!r} in sample: {s.get('key')}"
                            )
                        x = None
                    else:
                        x = _to_tensor(s[k])

                    flat_vals.append(x)
                    present.append(x is not None)
                    if x is not None:
                        if align is not None:
                            flat_lens.append(x.shape[-1])
                        if ref_tensor is None:
                            ref_tensor = x

            else:
                raise ValueError(f"[collate] Unknown axis: {axis}")

        # Sequence-like tensors use the final dimension as time.
        target_len = None
        if align is not None:
            if len(flat_lens) > 0:
                if align == "max":
                    target_len = max(flat_lens)
                elif align == "min":
                    target_len = min(flat_lens)
                else:
                    raise ValueError(f"[collate] Unknown align mode: {align}")
            else:
                if default_shape is None:
                    raise RuntimeError(
                        f"[collate] Cannot infer target length for {out_key!r}: "
                        "no real sample exists in batch and no default_shape is set."
                    )
                target_len = int(default_shape[-1])

        # Missing optional tensors are materialized just before stacking.
        out_feats = []
        for idx, x in enumerate(flat_vals):
            if x is None:
                if required:
                    raise RuntimeError(
                        f"[collate] Required feature {out_key!r} missing for batch sample {idx}"
                    )
                x = _fallback_tensor(ref_tensor, default_shape, fill_value,
                                     out_key)

            if target_len is not None:
                x = _pad_or_crop_to_len(x, target_len, fill_value, pad_mode)
            out_feats.append(x)

        new_batch[out_key] = torch.stack(out_feats, dim=0)
        if emit_present:
            new_batch[f"{out_key}_present"] = torch.tensor(present,
                                                           dtype=torch.bool)

    return new_batch
