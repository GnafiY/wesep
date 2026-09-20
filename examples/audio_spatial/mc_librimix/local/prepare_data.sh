#!/usr/bin/env bash

set -euo pipefail

dataset_root=
audio_cue_root=
data=data
mix_condition=reverb
target_condition=early

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

if [ -z "${dataset_root}" ] || [ -z "${audio_cue_root}" ]; then
  echo "--dataset-root and --audio-cue-root are required." >&2
  exit 1
fi

# Build one sample list and register both target-speaker cues.
for split in train-100 dev test; do
  output_dir=${data}/${split}
  spatial_resource=${dataset_root}/${split}/cues/spatial.json
  mix_dir=${dataset_root}/${split}/mix_${mix_condition}
  s1_dir=${dataset_root}/${split}/s1_${target_condition}
  s2_dir=${dataset_root}/${split}/s2_${target_condition}
  if [ "${split}" = "train-100" ]; then
    audio_resource=${audio_cue_root}/${split}/cues/audio.json
    audio_policy=random
    audio_key=spk_id
  else
    audio_resource=${audio_cue_root}/${split}/cues/fixed_enroll.json
    audio_policy=fixed
    audio_key=mix_spk_id
  fi

  if [ ! -f "${audio_resource}" ]; then
    echo "Audio cue index not found: ${audio_resource}" >&2
    exit 1
  fi
  if [ ! -f "${spatial_resource}" ]; then
    echo "Spatial cue index not found: ${spatial_resource}" >&2
    exit 1
  fi
  for audio_dir in "${mix_dir}" "${s1_dir}" "${s2_dir}"; do
    if [ ! -d "${audio_dir}" ]; then
      echo "Generated audio directory not found: ${audio_dir}" >&2
      exit 1
    fi
  done

  mkdir -p "${output_dir}"
  python local/scan_librimix.py \
    --mix-dir "${mix_dir}" \
    --s1-dir "${s1_dir}" \
    --s2-dir "${s2_dir}" \
    --outfile "${output_dir}/samples.jsonl"

  ln -sf samples.jsonl "${output_dir}/raw.list"
  audio_resource=$(realpath "${audio_resource}")
  spatial_resource=$(realpath "${spatial_resource}")
  cat > "${output_dir}/cues.yaml" <<EOF
cues:
  audio:
    type: wav
    format: waveform
    guaranteed: true
    policy:
      type: ${audio_policy}
      key: ${audio_key}
      resource: ${audio_resource}
  spatial:
    type: npy
    format: fields
    guaranteed: true
    fields: [azimuth, elevation]
    policy:
      type: fixed
      key: mix_spk_id
      resource: ${spatial_resource}
EOF
done
