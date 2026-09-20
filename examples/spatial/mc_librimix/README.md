# Spatial Multichannel LibriMix Recipe

> Last updated: 2026-09-21

This Linux recipe derives a fixed 16 kHz, four-channel spatial dataset from
the official Libri2Mix `wav16k/min` sources. The microphone geometry follows
the ReSpeaker USB four-microphone channel order used by this recipe.

The array is viewed from the front with `+x` pointing right and `+y` pointing
up: channels 1 to 4 are upper-right, upper-left, lower-left, and lower-right.
Azimuth zero points along `+x` and positive angles rotate counterclockwise.

## Inputs

The Libri2Mix root must contain aligned, gain-adjusted sources:

```text
wav16k/min/{train-100,dev,test}/{s1,s2}/*.wav
```

The simulation requires `numpy`, `scipy`, `soundfile`, `pyyaml`, and a working
Linux installation of `gpuRIR`.

## Stages

Stage 0 recreates the deterministic scenario manifest, generates early and
fully reverberant waveforms, and exports standard spatial NPY cues.
Stage 1 selects which generated waveform conditions to use and creates WeSep
`samples.jsonl`, `raw.list`, and `cues.yaml` files. It does not read the room
scenario manifest. Later stages create shards, train, infer, and score.

The complete sequence is simulation (Stage 0), list and cue preparation
(Stage 1), optional shard creation (Stage 2), training (Stage 3), checkpoint
averaging (Stage 4), inference (Stage 5), and scoring (Stage 6). Data
preparation and training are the validated v0.1 path; inference and scoring
remain under release-wide validation.

Run the data stages from this recipe directory:

```bash
./run.sh --stage 0 --stop_stage 1 \
  --librimix_root /path/to/Libri2Mix/wav16k/min \
  --mc_librimix_root /path/to/WeSep-MC-LibriMix/wav16k/min \
  --simulation_config confs/data/respeaker_4mic_16k.yaml \
  --mix_condition reverb \
  --target_condition reverb
```

The simulation config is exposed by `run.sh` because it defines the array,
room, source, RIR, and generated waveform settings. The deterministic scenario
manifest is created automatically under
`<mc_librimix_root>/metadata/scenarios_v1.csv.gz`. Stage 0 compares each
new manifest with the previous version: unchanged scenarios resume from
existing waveforms, while changed simulation settings rebuild the waveforms.

During Stage 1, `mix_condition` and `target_condition` independently select
`early` or `reverb`. For example, `early/early` compares a lightly reverberant
multichannel task, while `reverb/early` trains separation with dereverberation.
These choices affect the generated WeSep sample list, not the simulated
waveform directories.

Stage 0 writes generated waveform variants as:

```text
<mc_librimix_root>/
  train-100/{mix_early,mix_reverb,s1_early,s1_reverb,s2_early,s2_reverb,cues}/
  dev/{mix_early,mix_reverb,s1_early,s1_reverb,s2_early,s2_reverb,cues}/
  test/{mix_early,mix_reverb,s1_early,s1_reverb,s2_early,s2_reverb,cues}/
```

Each `cues/spatial.json` uses `mixture_id::speaker_id` keys and points to NPY
files containing named `azimuth`, `elevation`, and `distance` fields.

The current WeSep source reader uses channel 0 of each multichannel source as
the training target. Spatial models using this recipe must therefore keep
`separator.reference_channel: 0`.

An existing two-speaker, Libri2Mix-style multichannel dataset can reuse Stage 1
without a simulation manifest when it provides directories matching the selected
`mix_condition` and `target_condition`:

```bash
./run.sh --stage 1 --stop_stage 1 \
  --mc_librimix_root /path/to/standard/multichannel-dataset \
  --data data/custom \
  --mix_condition reverb \
  --target_condition reverb
```

Angles in the scenario CSV are stored in degrees for readability. Spatial NPY
cues are stored in radians because the WeSep spatial frontend consumes radians.

The generated waveform root can also be consumed by
`examples/audio/mc_librimix` to train a speaker-cue model without
enabling spatial features.
