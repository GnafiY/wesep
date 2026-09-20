#!/usr/bin/env bash

set -euo pipefail

voxceleb2_root=
output_root=
mixture_manifest=
num_workers=8
visual_frontend=raw_video
visual_frontend_checkpoint=
visual_device=cuda

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

if [ -z "${voxceleb2_root}" ] || [ -z "${output_root}" ]; then
  echo "Both --voxceleb2-root and --output-root are required." >&2
  exit 1
fi
case "${visual_frontend}" in
  raw_video|muse) ;;
  *)
    echo "Unsupported visual frontend: ${visual_frontend}. Supported: raw_video, muse." >&2
    exit 1
    ;;
esac
if [ "${visual_frontend}" != "raw_video" ] && [ -z "${visual_frontend_checkpoint}" ]; then
  echo "--visual-frontend-checkpoint is required for ${visual_frontend}." >&2
  exit 1
fi

# For exact MuSE sample and gain reproduction, download the official manifest
# and pass it through --mixture-manifest:
# wget -O /path/to/mixture_data_list_2mix.csv \
#   https://raw.githubusercontent.com/zexupan/MuSE/master/data/voxceleb2-800/mixture_data_list_2mix.csv
# local/simulate_data.sh \
#   --voxceleb2-root /path/to/VoxCeleb2/orig \
#   --output-root /path/to/VoxCeleb2Mix \
#   --mixture-manifest /path/to/mixture_data_list_2mix.csv
#
# Omit --mixture-manifest to create a deterministic custom list instead.

# Step 1: prepare source audio and select the fixed mixture manifest.
audio_root="${output_root}/audio_clean"
if [ -z "${mixture_manifest}" ]; then
  mixture_manifest="${output_root}/metadata/mixture_data_list_2mix.csv"

  # A custom manifest requires metadata for all candidate VoxCeleb2 clips.
  python local/simulation/extract_audio.py \
    --video-root "${voxceleb2_root}" \
    --audio-root "${audio_root}" \
    --num-workers "${num_workers}"
  python local/simulation/create_mixture_list.py \
    --audio-root "${audio_root}" \
    --output "${mixture_manifest}"
else
  # The official MuSE manifest can extract only the clips it references.
  python local/simulation/extract_audio.py \
    --video-root "${voxceleb2_root}" \
    --audio-root "${audio_root}" \
    --manifest "${mixture_manifest}" \
    --num-workers "${num_workers}"
fi

# Step 2: generate min-length audio and raw MP4 cue indexes.
python local/simulation/create_voxceleb2mix.py \
  --manifest "${mixture_manifest}" \
  --audio-root "${audio_root}" \
  --video-root "${voxceleb2_root}" \
  --output-root "${output_root}" \
  --num-workers "${num_workers}"

# Step 3: optionally replace raw MP4 cues with precomputed visual features.
if [ "${visual_frontend}" != "raw_video" ]; then
  bash local/extract_visual_features.sh \
    --frontend "${visual_frontend}" \
    --dataset-root "${output_root}" \
    --checkpoint "${visual_frontend_checkpoint}" \
    --device "${visual_device}"
fi
