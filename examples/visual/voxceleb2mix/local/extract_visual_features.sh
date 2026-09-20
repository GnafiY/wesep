#!/usr/bin/env bash

set -euo pipefail

frontend=
dataset_root=
checkpoint=
output_root=
device=cuda

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

if [ -z "${frontend}" ] || [ -z "${dataset_root}" ]; then
  echo "Both --frontend and --dataset-root are required." >&2
  exit 1
fi

# Dispatch each supported frontend to its own feature extractor.
case "${frontend}" in
  muse)
    # MuSE visual frontend checkpoint:
    # wget --no-check-certificate \
    #   'https://drive.google.com/uc?export=download&id=17BrWunfGl0T9QBw837wyQ88TFQ-dsX4K' \
    #   -O pretrain_networks/muse_visual_frontend.pt
    if [ -z "${checkpoint}" ]; then
      echo "--checkpoint is required for frontend=muse." >&2
      exit 1
    fi
    if [ -z "${output_root}" ]; then
      output_root="${dataset_root}/muse_frontend"
    fi
    python local/feature/extract_muse_frontend.py \
      --dataset-root "${dataset_root}" \
      --checkpoint "${checkpoint}" \
      --output-root "${output_root}" \
      --device "${device}"
    ;;
  *)
    echo "Unsupported visual frontend: ${frontend}. Supported: muse." >&2
    exit 1
    ;;
esac
