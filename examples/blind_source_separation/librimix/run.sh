#!/usr/bin/env bash

# Libri2Mix/Libri3Mix BSS recipe.
set -euo pipefail
. ./path.sh || exit 1

# Run no stage by default; select the stage range explicitly.
stage=-1
stop_stage=-1

# Stage 1: Libri2Mix or Libri3Mix audio indexes
librimix_root= # Existing Libri2Mix or Libri3Mix root containing wav16k/min
num_sources=  # CLI override, otherwise read from config
min_max=min
train_split=train-100
splits= # Optional subset, e.g. "test", or all available splits.
data=data
noise_type=clean
data_type=raw # First version uses raw data only.

# Training
gpus=
config=confs/bss_bsrnn.yaml
exp_dir=
checkpoint=
num_avg=
avg_epochs=

# Inference and scoring
fs=
infer_gpu= # Optional single-device override; otherwise use gpus.
infer_tag= # Optional output subdirectory, as in the textual recipe.
save_results=true
help_message='Run from examples/blind_source_separation/librimix.
Usage: ./run.sh --stage 1 --stop_stage 1 --librimix_root /path/to/Libri2Mix
Libri3Mix: add --num_sources 3 (uses the same config).
See README.md for all stages.'

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1
. "${WESEP_ROOT}/tools/resolve_config.sh" || exit 1

# Resolve CLI > YAML > recipe defaults. train.py propagates the source count
# into model and loss arguments after configuration merging.
num_sources=$(resolve_yaml_value "${num_sources}" "${config}" "num_sources" "2")
gpus=$(resolve_yaml_value "${gpus}" "${config}" "gpus" "[0]")
gpus=${gpus// /}
num_gpus=$(echo "${gpus}" | awk -F ',' '{print NF}')
exp_dir_override=${exp_dir}
exp_dir=$(resolve_yaml_value "${exp_dir}" "${config}" "exp_dir" "exp/BSS_BSRNN")
if [ -z "${exp_dir_override}" ] && [ "${num_sources}" = "3" ] && [ "${exp_dir}" = "exp/BSS_BSRNN" ]; then
  exp_dir=exp/BSS_BSRNN_3spk
fi
num_avg=$(resolve_yaml_value "${num_avg}" "${config}" "num_avg" "5")
fs=$(resolve_fs "${fs}" "${config}" "16k")
infer_tag=$(resolve_yaml_value "${infer_tag}" "${config}" "infer_tag" "")

if [[ ! "${num_sources}" =~ ^[23]$ ]]; then
  echo "--num_sources must be 2 or 3." >&2
  exit 1
fi
if [ "${data_type}" != "raw" ] || [ "${fs}" != "16k" ] || [ "${noise_type}" != "clean" ]; then
  echo "This recipe currently supports raw data, 16k and mix_clean." >&2
  exit 1
fi
# Keep two-source and three-source indexes in separate directories.
data=${data}/Libri${num_sources}Mix/${noise_type}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  if [ -z "${librimix_root}" ]; then
    echo "Stage 1 requires --librimix_root pointing to Libri2Mix or Libri3Mix." >&2
    exit 1
  fi
  echo "Stage 1: Build Libri${num_sources}Mix audio indexes"
  local/prepare_data.sh \
    --mix_data_path "${librimix_root}/wav${fs}/${min_max}" \
    --data "${data}" \
    --train_split "${train_split}" \
    --num_sources "${num_sources}" \
    ${splits:+--splits "${splits}"}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: Skip shards in this raw-only recipe"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: Train BSS"
  if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/latest_checkpoint.pt" ]; then
    checkpoint=${exp_dir}/models/latest_checkpoint.pt
  fi
  export OMP_NUM_THREADS=8
  torchrun --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
    "${WESEP_ROOT}/wesep/bin/train.py" \
    --config "${config}" \
    --task bss \
    --exp_dir "${exp_dir}" \
    --gpus "${gpus}" \
    --num_avg "${num_avg}" \
    --num_sources "${num_sources}" \
    --data_type "${data_type}" \
    --train_data "${data}/${train_split}/${data_type}.list" \
    --train_samples "${data}/${train_split}/samples.jsonl" \
    --val_data "${data}/dev/${data_type}.list" \
    --val_samples "${data}/dev/samples.jsonl" \
    ${checkpoint:+--checkpoint "${checkpoint}"}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: Average checkpoints"
  avg_mode=final
  if [ -n "${avg_epochs}" ]; then
    avg_mode=best
  fi
  python "${WESEP_ROOT}/wesep/bin/average_model.py" \
    --dst_model "${exp_dir}/models/avg_model.pt" \
    --src_path "${exp_dir}/models" \
    --mode "${avg_mode}" \
    --num "${num_avg}" \
    ${avg_epochs:+--epochs "${avg_epochs}"}
  checkpoint=${exp_dir}/models/avg_model.pt
fi

if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/avg_model.pt" ]; then
  checkpoint=${exp_dir}/models/avg_model.pt
fi
infer_exp_dir=${exp_dir}
if [ -n "${infer_tag}" ]; then
  infer_exp_dir=${exp_dir}/inference/${infer_tag}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  if [ -z "${checkpoint}" ]; then
    echo "Stage 5 requires --checkpoint or ${exp_dir}/models/avg_model.pt." >&2
    exit 1
  fi
  echo "Stage 5: Infer all sources"
  # Prefer the exact model configuration saved during training.
  if [ -f "${exp_dir}/config.yaml" ]; then
    config=${exp_dir}/config.yaml
  fi
  echo "Inference output: ${infer_exp_dir}"
  gpu_list=${gpus#[}
  gpu_list=${gpu_list%]}
  IFS=',' read -r -a infer_gpus <<< "${gpu_list}"
  if [ -n "${infer_gpu}" ]; then
    infer_gpus=("${infer_gpu}")
  fi
  num_infer_gpus=${#infer_gpus[@]}

  if [ "${num_infer_gpus}" -eq 1 ]; then
    python "${WESEP_ROOT}/wesep/bin/infer.py" \
      --config "${config}" \
      --task bss \
      --fs "${fs}" \
      --gpus "${infer_gpus[0]}" \
      --exp_dir "${exp_dir}" \
      --num_sources "${num_sources}" \
      --data_type "${data_type}" \
      --test_data "${data}/test/${data_type}.list" \
      --test_samples "${data}/test/samples.jsonl" \
      --save_wav "${save_results}" \
      --rank 0 \
      --infer_tag "${infer_tag}" \
      --checkpoint "${checkpoint}"
  else
    test_list=${data}/test/${data_type}.list
    split_dir=${infer_exp_dir}/split${num_infer_gpus}
    mkdir -p "${split_dir}"
    split_lists=()
    for i in "${!infer_gpus[@]}"; do
      split_lists+=("${split_dir}/${i}.${data_type}.list")
    done
    perl "${WESEP_ROOT}/tools/split_scp.pl" "${test_list}" "${split_lists[@]}"

    pids=()
    for i in "${!infer_gpus[@]}"; do
      job_dir=${infer_exp_dir}/jobs/${i}
      mkdir -p "${job_dir}"
      python "${WESEP_ROOT}/wesep/bin/infer.py" --config "${config}" \
        --task bss --fs "${fs}" --gpus "${infer_gpus[$i]}" \
        --exp_dir "${job_dir}" --log_dir "${infer_exp_dir}" \
        --num_sources "${num_sources}" \
        --data_type "${data_type}" --test_data "${split_lists[$i]}" \
        --test_samples "${data}/test/samples.jsonl" \
        --save_wav "${save_results}" --rank "$i" \
        --infer_tag "" --checkpoint "${checkpoint}" \
        > "${job_dir}/infer_launcher.log" 2>&1 &
      pids+=("$!")
    done
    trap 'kill "${pids[@]}" 2>/dev/null || true; exit 1' INT TERM
    infer_failed=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        infer_failed=1
      fi
    done
    trap - INT TERM
    if [ "${infer_failed}" -ne 0 ]; then
      echo "Inference failed; see ${infer_exp_dir}/jobs/*/infer_launcher.log." >&2
      exit 1
    fi

    if [ "${save_results}" = "true" ]; then
      mkdir -p "${infer_exp_dir}/audio"
      sort "${data}/test/keys" > "${split_dir}/expected.keys"
      for source in $(seq 1 "${num_sources}"); do
        source_scps=()
        for i in "${!infer_gpus[@]}"; do
          source_scps+=("${infer_exp_dir}/jobs/${i}/audio/spk${source}.scp")
        done
        sort -k1,1 "${source_scps[@]}" > "${infer_exp_dir}/audio/spk${source}.scp"
        awk '{print $1}' "${infer_exp_dir}/audio/spk${source}.scp" > "${split_dir}/actual.keys"
        cmp "${split_dir}/expected.keys" "${split_dir}/actual.keys"
      done
    fi
  fi
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: SI-SDR PIT scoring"
  source_args=()
  for i in $(seq 1 "${num_sources}"); do
    source_args+=(--ref_scp "${data}/test/ref${i}.scp")
    source_args+=(--inf_scp "${infer_exp_dir}/audio/spk${i}.scp")
  done
  python "${WESEP_ROOT}/wesep/bin/score.py" \
    --task bss \
    --key_file "${data}/test/keys" \
    --mixture_scp "${data}/test/wav.scp" \
    "${source_args[@]}" \
    --output_dir "${infer_exp_dir}/scores/test"
fi
