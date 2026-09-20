# Multichannel LibriMix Speaker-Cue Recipe

> Last updated: 2026-09-21

This Linux recipe trains the speaker-cue BSRNN on the same four-channel
LibriMix waveforms used by `examples/spatial/mc_librimix`. It does
not use spatial features and does not simulate the dataset again.

Prepare the mono Libri2Mix speaker cues first with
`examples/audio/librimix/run.sh`. Use an absolute Libri2Mix path there so the
generated enrollment indexes remain valid from other recipe directories.

Stage 1 scans the multichannel `mix/s1/s2` view and joins it with the existing
mono enrollment indexes by LibriMix speaker and mixture keys:

```bash
./run.sh --stage 1 --stop_stage 1 \
  --mc_librimix_root /path/to/WeSep-MC-LibriMix/wav16k/min \
  --audio_cue_root ../librimix/data/clean
```

The multichannel dataset must contain:

```text
<mc_librimix_root>/{train-100,dev,test}/{mix,s1,s2}/*.wav
```

Training uses random enrollment utterances indexed by `spk_id`. Development
and test use fixed enrollment utterances indexed by `mix_key::spk_id`. The
mixture IDs and speaker IDs must therefore match the original Libri2Mix data.

The config enables only speaker features supported by multichannel mixtures.
The enrollment signal remains single-channel, while the separator consumes a
four-channel mixture and reconstructs `reference_channel: 0`.

## Stages

Stages 1–6 build lists, optionally create shards, train, average checkpoints,
infer, and score, respectively. Generate the multichannel waveforms with the
spatial recipe before Stage 1. Data preparation and training are the validated
v0.1 path; inference and scoring remain under release-wide validation.
