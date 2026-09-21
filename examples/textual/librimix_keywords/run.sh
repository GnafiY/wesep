#!/usr/bin/env bash

# Copyright 2026 Ke Zhang (kylezhang1118@gmail.com)
#           2026 Haoyu Li (haoyu.li.cs@sjtu.edu.cn)

set -euo pipefail
. ./path.sh || exit 1

# Run no stage by default; select the stage range explicitly.
stage=-1
stop_stage=-1

# Stage 0: official DAE-KCE model and text resources
kce_resource_dir=exp/kce

# Stage 1: Libri2Mix audio and Nemo-transcript keyword cue indexes
librimix_root=       # Existing Libri2Mix root containing wav16k/min
data=data
noise_type=clean
dev_keyword_count=4
test_keyword_count=4 # Official DAE test condition: 1, 2, 3, or 4 words

# Training data format
data_type=raw # shard/raw

# Training
gpus=
config=confs/tse_bsrnn_textual.yaml
exp_dir=exp/TSE_BSRNN_TEXTUAL_KEYWORDS
checkpoint=
num_avg=

# Inference and scoring
fs=
save_results=true
use_pesq=true
use_dnsmos=true
dnsmos_use_gpu=true

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1
. "${WESEP_ROOT}/tools/resolve_config.sh" || exit 1

# Resolve values with explicit CLI/submit overrides first, then config, then
# recipe fallback defaults.
gpus=$(resolve_yaml_value "${gpus}" "${config}" "gpus" "[0]")
gpus=${gpus// /}
num_gpus=$(echo "${gpus}" | awk -F ',' '{print NF}')
exp_dir=$(resolve_yaml_value "${exp_dir}" "${config}" "exp_dir" "exp/TSE_BSRNN_TEXTUAL_KEYWORDS")
num_avg=$(resolve_yaml_value "${num_avg}" "${config}" "num_avg" "5")
fs=$(resolve_fs "${fs}" "${config}" "16k")

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: Download official DAE-KCE resources"
  local/dae_kce/download_resources.sh \
    --output-dir "${kce_resource_dir}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Build WeSep audio and textual cue indexes"
  local/prepare_data.sh \
    --mix-data-path "${librimix_root}/wav16k/min" \
    --resource-dir "${kce_resource_dir}" \
    --data "${data}" \
    --noise-type "${noise_type}" \
    --dev-keyword-count "${dev_keyword_count}" \
    --test-keyword-count "${test_keyword_count}"
fi

data=${data}/${noise_type}

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ "${data_type}" = "shard" ]; then
  echo "Stage 2: Create shards"
  for split in train-100 dev test; do
    python "${WESEP_ROOT}/tools/make_shards_from_samples.py" \
      --samples "${data}/${split}/samples.jsonl" \
      --num_utts_per_shard 1000 \
      --num_threads 16 \
      --prefix shards \
      --shuffle \
      "${data}/${split}/shards" \
      "${data}/${split}/shard.list"
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: Train"
  if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/latest_checkpoint.pt" ]; then
    checkpoint=${exp_dir}/models/latest_checkpoint.pt
  fi
  export OMP_NUM_THREADS=8
  torchrun --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
    "${WESEP_ROOT}/wesep/bin/train.py" \
    --config "${config}" \
    --exp_dir "${exp_dir}" \
    --gpus "${gpus}" \
    --num_avg "${num_avg}" \
    --data_type "${data_type}" \
    --train_data "${data}/train-100/${data_type}.list" \
    --train_cues "${data}/train-100/cues.yaml" \
    --train_samples "${data}/train-100/samples.jsonl" \
    --val_data "${data}/dev/${data_type}.list" \
    --val_cues "${data}/dev/cues.yaml" \
    --val_samples "${data}/dev/samples.jsonl" \
    ${checkpoint:+--checkpoint "${checkpoint}"}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: Average checkpoints"
  # This stage runs only when explicitly selected; edit the epochs below.
  python "${WESEP_ROOT}/wesep/bin/average_model.py" \
    --dst_model "${exp_dir}/models/avg_best_model.pt" \
    --src_path "${exp_dir}/models" \
    --mode best \
    --epochs "138,141"
fi

if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/avg_best_model.pt" ]; then
  checkpoint=${exp_dir}/models/avg_best_model.pt
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: Infer"

  if [ "${num_gpus}" -eq 1 ]; then
    infer_gpu="${gpus#[}"
    infer_gpu="${infer_gpu%]}"
    python "${WESEP_ROOT}/wesep/bin/infer.py" \
      --config "${config}" \
      --fs "${fs}" \
      --gpus "${infer_gpu}" \
      --exp_dir "${exp_dir}" \
      --data_type "${data_type}" \
      --test_data "${data}/test/${data_type}.list" \
      --test_cues "${data}/test/cues.yaml" \
      --test_samples "${data}/test/samples.jsonl" \
      --save_wav "${save_results}" \
      ${checkpoint:+--checkpoint "${checkpoint}"}
  else
    # Keep the same GPU-list convention as training (for example "[0,1,2,3]").
    gpu_list="${gpus#[}"
    gpu_list="${gpu_list%]}"
    IFS=',' read -r -a infer_gpus <<< "${gpu_list}"

    test_list=${data}/test/${data_type}.list
    split_dir=${data}/test/split${num_gpus}
    mkdir -p "${split_dir}"
    split_lists=()
    for i in "${!infer_gpus[@]}"; do
      split_lists+=("${split_dir}/${i}.${data_type}.list")
    done
    perl "${WESEP_ROOT}/tools/split_scp.pl" "${test_list}" "${split_lists[@]}"

    # Workers share the audio directory, but use independent logs and defer scp
    # creation until every worker has completed, preventing concurrent rewrites.
    pids=()
    for i in "${!infer_gpus[@]}"; do
      python "${WESEP_ROOT}/wesep/bin/infer.py" --config "${config}" \
        --fs "${fs}" --gpus "${infer_gpus[$i]}" --exp_dir "${exp_dir}" \
        --data_type "${data_type}" \
        --test_data "${split_lists[$i]}" \
        --test_cues "${data}/test/cues.yaml" \
        --test_samples "${data}/test/samples.jsonl" \
        --save_wav "${save_results}" \
        --write_enhanced_scp false \
        ${checkpoint:+--checkpoint "$checkpoint"} &
      pids+=("$!")
    done
    trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    trap - INT TERM

    if [[ "${save_results}" == "true" ]]; then
      python -c "from wesep.utils.utils import generate_enahnced_scp; generate_enahnced_scp(r'${exp_dir}/audio', extension='wav')"
    fi
  fi
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: Score"
  python "${WESEP_ROOT}/tools/build_tse_reference_scp.py" \
    --samples "${data}/test/samples.jsonl" \
    --output "${data}/test/single.wav.scp" \
    --inference-scp "${exp_dir}/audio/spk1.scp" \
    --check-source-files
  "${WESEP_ROOT}/tools/score.sh" \
    --dset "${data}/test" \
    --exp_dir "${exp_dir}" \
    --fs "${fs}" \
    --use_pesq "${use_pesq}" \
    --use_dnsmos "${use_dnsmos}" \
    --dnsmos_use_gpu "${dnsmos_use_gpu}" \
    --n_gpu "${num_gpus}"
fi
