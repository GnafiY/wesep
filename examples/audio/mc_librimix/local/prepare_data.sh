#!/usr/bin/env bash

set -euo pipefail

dataset_root=
audio_cue_root=
data=data

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

if [ -z "${dataset_root}" ] || [ -z "${audio_cue_root}" ]; then
  echo "--dataset-root and --audio-cue-root are required." >&2
  exit 1
fi

# Scan multichannel mixtures and join the existing mono enrollment indexes.
for split in train-100 dev test; do
  output_dir=${data}/${split}
  if [ "${split}" = "train-100" ]; then
    cue_resource=${audio_cue_root}/${split}/cues/audio.json
  else
    cue_resource=${audio_cue_root}/${split}/cues/fixed_enroll.json
  fi

  if [ ! -f "${cue_resource}" ]; then
    echo "Audio cue index not found: ${cue_resource}" >&2
    exit 1
  fi

  mkdir -p "${output_dir}"
  python local/scan_librimix.py \
    --dataset-dir "${dataset_root}/${split}" \
    --outfile "${output_dir}/samples.jsonl"

  ln -sf samples.jsonl "${output_dir}/raw.list"
  cue_resource=$(realpath "${cue_resource}")
  policy=fixed
  key=mix_spk_id
  if [ "${split}" = "train-100" ]; then
    policy=random
    key=spk_id
  fi

  cat > "${output_dir}/cues.yaml" <<EOF
cues:
  audio:
    type: wav
    format: waveform
    guaranteed: true
    policy:
      type: ${policy}
      key: ${key}
      resource: ${cue_resource}
EOF
done
