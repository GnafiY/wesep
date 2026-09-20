#!/usr/bin/env bash

set -euo pipefail

dataset_root=
data=data

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

if [ -z "${dataset_root}" ]; then
  echo "--dataset-root is required." >&2
  exit 1
fi

# Read the active raw-video or precomputed-feature view selected by Stage 0.
cue_description="${dataset_root}/visual_cue.json"
if [ ! -f "${cue_description}" ]; then
  echo "Visual cue description not found: ${cue_description}" >&2
  exit 1
fi
cue_info=$(python -c \
  'import json, sys; x=json.load(open(sys.argv[1])); print(x["type"], x["format"], x["index"], sep="\t")' \
  "${cue_description}")
IFS=$'\t' read -r cue_type cue_format cue_index_name <<< "${cue_info}"

# Build WeSep samples and cue configurations from one dataset root.
for split in train val test; do
  output_dir="${data}/${split}"
  cue_resource="${dataset_root}/${split}/cues/${cue_index_name}"
  if [ ! -f "${cue_resource}" ]; then
    echo "Visual cue index not found: ${cue_resource}" >&2
    exit 1
  fi

  python local/scan_voxceleb2mix.py \
    --dataset-dir "${dataset_root}/${split}" \
    --cue-index "${cue_resource}" \
    --outfile "${output_dir}/samples.jsonl"

  ln -sf samples.jsonl "${output_dir}/raw.list"
  cat > "${output_dir}/cues.yaml" <<EOF
cues:
  visual:
    # Stage 0 currently produces mp4/raw_video or npy/muse_frontend.
    type: ${cue_type}
    format: ${cue_format}
    guaranteed: true
    policy:
      type: fixed
      key: mix_spk_id
      resource: ${cue_resource}
EOF
done
