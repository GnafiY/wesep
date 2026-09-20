# VoxCeleb1 Online-Mix Speaker-Cue Recipe

> Last updated: 2026-09-21

This recipe dynamically mixes single-speaker VoxCeleb1 utterances for training.
Online mixing is training-only; development and test always use fixed Libri2Mix
mixtures and enrollment mappings.

Inputs:

```text
<VoxCeleb1>/wav/<speaker>/<video>/<utterance>.wav
<Libri2Mix>/wav16k/min/{dev,test}/{mix_clean,s1,s2}/*.wav
```

Prepare the online training pool and fixed evaluation lists:

```bash
./run_online.sh --stage 1 --stop_stage 1 \
  --Vox1_dir /path/to/VoxCeleb1/wav \
  --Libri2Mix_dir /path/to/Libri2Mix
```

The default config uses `whole_utt: false`, creates fixed-length chunks, and
applies timeline patterns only to the generated audio mixture and target. It
currently supports single-channel online mixing. Stage 2 uses the dedicated
online shard writer for `train-vox1` and the standard writer for `dev/test`.

## Stages

Stages 1–5 prepare data, create the default shard inputs, train, average
checkpoints, and run recipe inference, respectively. Data preparation and
training are the validated v0.1 path; inference remains under release-wide
validation.
