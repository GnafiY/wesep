# Cue and Collate Contract

> Last updated: 2026-09-21
> Language: English | [中文](cue_collate_design.zh-CN.md)

This document defines the boundary between cue metadata, cue processors,
collation, and model frontends in the current implementation.

## Responsibilities

- `cues.yaml` describes what a dataset split contains and how a cue is found.
- A modality processor loads the resource and converts it to a tensor with a
  documented layout.
- `build_collect_keys` selects the fields required by an experiment.
- `tse_collate_fn` expands the speaker axis, aligns the last dimension,
  materializes optional fallbacks, and stacks tensors.
- A model frontend interprets cue meaning and decides how missing cues affect
  computation.

Collate does not interpret waveform, video, direction, or text semantics.

## Split-Level Cue Schema

```yaml
cues:
  spatial:
    type: npy
    format: fields
    scope: speaker
    guaranteed: true
    fields: [azimuth, elevation]
    policy:
      type: fixed
      key: mix_spk_id
      resource: data/train/cues/spatial.json
    collate:
      default_shape: [2]
      fill_value: -999.0
```

Field meanings:

- `type`: physical payload or reader type.
- `format`: semantic representation produced by the reader.
- `scope`: currently only `speaker`.
- `guaranteed`: whether every target in the split has the cue.
- `policy.type`: `random` or `fixed`, as supported by that modality.
- `policy.key`: `spk_id` or `mix_spk_id`.
- `policy.resource`: JSON lookup table.
- `collate.default_shape`: fallback shape when an entire batch lacks the cue.
- `collate.fill_value`: value used for missing cues and constant padding.

## Implemented Cue Formats

The registry in `wesep/dataset/cues.py` accepts:

| Modality | Type | Format | Sample field |
| --- | --- | --- | --- |
| Audio | `wav` | `waveform` | `audio_spk{i}` |
| Audio | `npy` | `spk_embedding` | `audio_spk{i}` |
| Visual | `mp4` | `raw_video` | `visual_spk{i}` |
| Visual | `npy` | `muse_frontend` | `visual_spk{i}` |
| Spatial | `npy` | `fields` | `spatial_spk{i}` |
| Textual | `json` | `dae_keyword_phoneme` | `textual_spk{i}` |

The precomputed `muse_frontend` representation remains an experimental
compatibility path. It is not advertised as an official WeSep model.

## Experiment-Level Selection

The model experiment declares which dataset cues it consumes:

```yaml
dataset_args:
  cues:
    audio:
      use: true
      required: true
    spatial:
      use: true
      required: false
      default_shape: [2]
```

`use` enables the processor and collate field. `required` controls whether
every expanded target must contain the cue. A required experiment cue cannot be
paired with `guaranteed: false` in the split metadata.

## Collect-Spec Precedence

The effective collate specification is merged in this order:

```text
BASE_COLLECT_KEYS
  < cues.yaml cue.collate
  < dataset_args.cues.<modality>
  < dataset_args.collate.<field>
```

The final experiment override is intended for explicit field-level changes,
for example validation waveform alignment.

## Speaker-Axis Expansion

One mixture with `N` target speakers becomes `N` TSE batch items.

- Mix-axis tensors such as `wav_mix` are repeated for every target.
- Speaker-axis tensors use `wav_spk{i}` or `{modality}_spk{i}`.
- Metadata such as `key` is repeated; `spk` selects the current target.
- The output target field is named `wav_target`.

## Alignment and Missing Cues

The last tensor dimension is treated as time only when a collect spec declares
`align`.

| Field | Default alignment | Padding |
| --- | --- | --- |
| `wav_mix` | max | constant zero |
| `wav_target` | max | constant zero |
| `audio_aux` | min | crop |
| `spatial_aux` | max | edge |
| `visual_aux` | max | edge |
| `textual_aux` | max | constant zero |

Missing optional cues use the shape and dtype of a real cue in the same batch.
If the whole batch is missing that cue, `default_shape` is required.
`audio_aux_present`, `spatial_aux_present`, `visual_aux_present`, and
`textual_aux_present` are emitted as boolean tensors when enabled.

Fallback values are structural. A frontend must use the presence mask when a
numeric fallback could otherwise be interpreted as a real cue.

## Time Synchronization

- Waveform cues are resampled by their reader.
- Raw video is aligned to source-audio duration and then cropped by chunk
  metadata.
- Precomputed visual and dynamic spatial features treat the final dimension as
  time and may use `duration_sec` from the resource item.
- Textual phoneme IDs are discrete sequences and use zero padding.
- Collate does not infer physical frame rates.

## Current Boundaries

- Cue scope is speaker-level only.
- Non-time dimensions must match within a batch.
- Visual and spatial resources use fixed selection.
- Audio resources support fixed and random selection.
- Textual resources currently implement the DAE keyword-phoneme format.
- Missing-cue support is fully meaningful only in models that consume the
  corresponding `*_present` mask.
