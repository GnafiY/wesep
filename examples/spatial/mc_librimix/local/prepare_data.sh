#!/usr/bin/env bash

set -euo pipefail

dataset_root=
data=data
mix_condition=reverb
target_condition=early

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

if [ -z "${dataset_root}" ]; then
  echo "--dataset-root is required." >&2
  exit 1
fi

# Build WeSep lists from a selected generated waveform condition.
for split in train-100 dev test; do
  output_dir=${data}/${split}
  cue_resource=${dataset_root}/${split}/cues/spatial.json
  mix_dir=${dataset_root}/${split}/mix_${mix_condition}
  s1_dir=${dataset_root}/${split}/s1_${target_condition}
  s2_dir=${dataset_root}/${split}/s2_${target_condition}
  mkdir -p "${output_dir}"
  if [ ! -f "${cue_resource}" ]; then
    echo "Spatial cue index not found: ${cue_resource}" >&2
    exit 1
  fi
  for audio_dir in "${mix_dir}" "${s1_dir}" "${s2_dir}"; do
    if [ ! -d "${audio_dir}" ]; then
      echo "Generated audio directory not found: ${audio_dir}" >&2
      exit 1
    fi
  done

  python local/scan_librimix.py \
    --mix-dir "${mix_dir}" \
    --s1-dir "${s1_dir}" \
    --s2-dir "${s2_dir}" \
    --outfile "${output_dir}/samples.jsonl"

  ln -sf samples.jsonl "${output_dir}/raw.list"
  cat > "${output_dir}/cues.yaml" <<EOF
cues:
  spatial:
    type: npy
    format: fields
    guaranteed: true
    fields: [azimuth, elevation]
    policy:
      type: fixed
      key: mix_spk_id
      resource: ${cue_resource}
EOF
done
