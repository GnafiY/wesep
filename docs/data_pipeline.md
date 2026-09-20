# WeSep Data Pipeline Contract

> Last updated: 2026-09-21
> Language: English | [中文](data_pipeline.zh-CN.md)

This document describes the behavior implemented by `wesep.dataset`. It is a
code contract for contributors, not a list of future ideas.

## Pipeline Overview

`Dataset()` builds an `IterableDataset` in the following order:

```text
DataList
  -> source reader (raw JSONL or tar shard)
  -> waveform resampling and training chunking
  -> optional online mixing and augmentation
  -> modality-specific cue processors
  -> optional target expansion for evaluation
  -> optional speaker labels
  -> DataLoader
  -> build_collect_keys
  -> tse_collate_fn
  -> model batch
```

Online mixing and augmentation run only when `state == "train"`. Validation
and test data use fixed mixtures and retain complete utterances.

## Source Data

### Raw data

Each line in `raw.list` is a JSON object:

```json
{
  "key": "mix_key",
  "spk": ["speaker_a", "speaker_b"],
  "mix": {"default": ["/path/to/mix.wav"]},
  "src": {
    "speaker_a": ["/path/to/target_a.wav"],
    "speaker_b": ["/path/to/target_b.wav"]
  }
}
```

`spk` defines target order. Mixture files are concatenated along the channel
axis. Each target currently uses the first channel of its first source file.
The source reader applies a strict length and sample-rate policy.

### Shard data

Offline shards contain contiguous members for each sample:

```text
{key}.spk1
{key}.spk2
{key}.{audio_ext}
{key}_spk1.{audio_ext}
{key}_spk2.{audio_ext}
```

Shards preserve the same sample semantics as raw JSONL. They change storage and
I/O behavior, not the model contract.

### Distributed iteration

The data list is shuffled reproducibly by epoch, partitioned by distributed
rank, and then partitioned by DataLoader worker. When training shards are fewer
than workers, an empty worker selects one deterministic fallback shard. This
can repeat training data but prevents the worker from becoming inactive.

## Waveform Processing

Waveforms use channel-first layout `[C, T]`.

- Every state is resampled to `dataset_args.resample_rate`.
- With `whole_utt: false`, training uses `random_chunk`.
- Short training utterances follow `chunk_short_policy`, which defaults to
  right zero-padding.
- With `whole_utt: true`, training retains the complete utterance and may use
  `filter_len` to reject extreme durations.
- Validation and test do not use random chunking or training augmentation.

Chunk metadata is stored as a ratio so time-varying visual and spatial cues can
select the corresponding interval.

## Online Mixing

Online mixing accepts single-speaker raw data or online shards during training:

```text
single-speaker source
  -> sample_speaker_group
  -> apply_timeline
  -> optional single-channel reverb
  -> SNR mixing
  -> optional mixture noise
```

The current online path is single-channel. Evaluation must use fixed offline
mixtures. `source_key_spk{i}` preserves the original utterance key so
`mix_spk_id` cues can still be retrieved after dynamic grouping.

## Cue Configuration

The experiment config selects cues:

```yaml
dataset_args:
  cues:
    audio:
      use: true
      required: true
```

The split-level `cues.yaml` describes the available resource:

```yaml
cues:
  audio:
    type: wav
    format: waveform
    scope: speaker
    guaranteed: true
    policy:
      type: random
      key: spk_id
      resource: data/train/cues/audio.json
```

Implemented representations are:

| Cue | Type and format | Processor output | Policy |
| --- | --- | --- | --- |
| Audio | `wav / waveform` | `[C, T]` | random or fixed |
| Audio | `npy / spk_embedding` | `[D]` | random or fixed |
| Visual | `mp4 / raw_video` | `[H, W, C, T]` | fixed |
| Visual | `npy / muse_frontend` | `[..., T]` | fixed, experimental compatibility path |
| Spatial | `npy / fields` | `[F]` or `[F, T]` | fixed |
| Textual | `json / dae_keyword_phoneme` | `[L]` phoneme IDs | fixed |

All current cues use `scope: speaker`. Lookup keys are:

- `spk_id`: `sample["spk{i}"]`
- offline `mix_spk_id`: `sample["key"] + "::" + sample["spk{i}"]`
- online `mix_spk_id`: `source_key_spk{i} + "::" + sample["spk{i}"]`

Resource JSON files are loaded lazily and cached in each DataLoader worker.

## Collation

`build_collect_keys` combines the base field definitions, split-level cue
defaults, and experiment overrides. `tse_collate_fn` then expands every
mixture into one item per target speaker:

```text
input mixtures: B_in
output targets: B_out = sum(num_speaker)
```

Core batch fields are:

```text
wav_mix:       FloatTensor [B_out, C_mix, T]
wav_target:    FloatTensor [B_out, 1, T]
spk:           list[str]
key:           list[str]
num_speaker:   list[int]
speaker_label: optional LongTensor [B_out]
audio_aux:     optional cue tensor
spatial_aux:   optional cue tensor
visual_aux:    optional cue tensor
textual_aux:   optional cue tensor
*_present:     optional BoolTensor [B_out]
```

Waveforms and most time-varying cues align to the longest item. Enrollment
waveforms align to the shortest enrollment in the batch. Missing optional cues
receive configured fallback tensors, and `*_present` records whether the
original cue existed.

## Current Boundaries

- Raw and shard inputs use the same strict waveform contract.
- Cue scope is currently speaker-level only.
- Online reverb, SNR mixing, and mixture noise are single-channel operations.
- Non-time dimensions must already match within a batch.
- Collate aligns lengths but does not resample cue frame rates.
- Missing-cue tensors provide structure; model frontends decide their semantic
  treatment.

Use `tools/test_dataset.py --config <config>` to inspect one model-ready batch
before launching a full training run.
