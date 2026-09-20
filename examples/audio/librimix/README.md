# LibriMix Speaker-Cue Recipe

> Last updated: 2026-09-21

This is the baseline single-channel speaker-cued TSE recipe. It scans official
Libri2Mix `wav16k/min` mixtures, builds random enrollment indexes for training,
and uses the fixed BUT SpeakerBeam enrollment mapping for development and test.

The expected input is:

```text
<Libri2Mix>/wav16k/min/{train-100,dev,test}/
  mix_clean/*.wav
  s1/*.wav
  s2/*.wav
```

Build the WeSep lists and download the configured WeSpeaker encoders:

```bash
./run.sh --stage 1 --stop_stage 1 \
  --Libri2Mix_dir /path/to/Libri2Mix
```

Stage 1 creates `data/clean/<split>/samples.jsonl`, `raw.list`, cue indexes,
and `cues.yaml`. Training cues are selected by `spk_id`; development and test
cues use `mix_key::spk_id`. The default config enables USEF and context, so its
speaker-encoder checkpoint path must exist before training.

Stage 2 creates shards only when `data_type=shard`; `data_type=raw` reads the
generated JSONL lists directly.

## Stages

| Stage | Action |
| --- | --- |
| 1 | Prepare lists, cue indexes, and configured speaker encoders |
| 2 | Create shards when `data_type=shard` |
| 3 | Train |
| 4 | Average selected checkpoints |
| 5 | Run recipe inference |
| 6 | Score the separated signals |

Data preparation and training are the validated v0.1 path. The inference and
scoring stages are implemented but remain under release-wide validation.
