#!/usr/bin/env bash
# Generate two/three-source WeSep manifests with shell only.
set -euo pipefail
shopt -s nullglob

if [ -z "${WESEP_ROOT:-}" ]; then
  . "$(dirname "${BASH_SOURCE[0]}")/../path.sh"
fi

mix_data_path=
data=
train_split=train-100
splits= # Optional space-separated subset, e.g. "test" or "train-100 train-360 dev test".
num_sources=2

help_message='Usage: local/prepare_data.sh --mix_data_path ROOT/wav16k/min --data OUTPUT --num_sources 2|3 [--splits "train-100 train-360 dev test"]'

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -n ${mix_data_path} ]] || die 'Set --mix_data_path to Libri2Mix or Libri3Mix/wav16k/min or max'
[[ ${num_sources} == 2 || ${num_sources} == 3 ]] || die 'num_sources must be 2 or 3'
data=${data:-data/Libri${num_sources}Mix/clean}
[[ ${train_split} == train-100 || ${train_split} == train-360 ]] || die 'Unsupported train_split'
IFS=' ' read -r -a data_splits <<< "${splits:-${train_split} dev test}"
for split in "${data_splits[@]}"; do
  [[ ${split} == train-100 || ${split} == train-360 || ${split} == dev || ${split} == test ]] || die "Unsupported split: ${split}"
done
mix_data_path=$(cd "${mix_data_path}" && pwd)
[[ ! ${mix_data_path} =~ [[:cntrl:]] ]] || die 'Control characters are not supported in dataset paths'

read_speakers() {
  local key=$1 utterance speaker previous
  local -a utterance_ids
  [[ ! ${key} =~ [[:space:]] ]] || die "Invalid LibriMix key: ${key}"
  IFS='_' read -r -a utterance_ids <<< "${key}"
  ((${#utterance_ids[@]} == num_sources)) || die "Expected ${num_sources} utterances in ${key}"
  spk_ids=()
  for utterance in "${utterance_ids[@]}"; do
    [[ ${utterance} =~ ^[0-9]+-[0-9]+-[0-9]+$ ]] || die "Invalid LibriSpeech utterance: ${utterance}"
    speaker=${utterance%%-*}
    for previous in "${spk_ids[@]}"; do
      [[ ${speaker} != "${previous}" ]] || die "Duplicate speaker ${speaker} in ${key}"
    done
    spk_ids+=("${speaker}")
  done
}

json_string() {
  # Control characters have been rejected; escape JSON backslashes and quotes.
  local escaped=$1
  escaped=${escaped//\\/\\\\}
  escaped=${escaped//\"/\\\"}
  printf '"%s"' "${escaped}"
}

# Check every split before producing indexes. Audio rate/length checks belong
# to the existing strict source parser and the BSS data validation step.
for split in "${data_splits[@]}"; do
  mix_dir="${mix_data_path}/${split}/mix_clean"
  mix_wavs=("${mix_dir}"/*.wav)
  ((${#mix_wavs[@]} > 0)) || die "No wav files in ${mix_dir}"
  for mix_wav in "${mix_wavs[@]}"; do
    filename=${mix_wav##*/}
    key=${filename%.wav}
    [[ -f ${mix_wav} ]] || die "Not a regular file: ${mix_wav}"
    read_speakers "${key}"
    for ((source_index=1; source_index<=num_sources; source_index++)); do
      [[ -f ${mix_data_path}/${split}/s${source_index}/${filename} ]] || die "Missing s${source_index} for ${key}"
    done
  done
done

for split in "${data_splits[@]}"; do
  output_dir="${data}/${split}"
  mix_dir="${mix_data_path}/${split}/mix_clean"
  mkdir -p "${output_dir}"
  : > "${output_dir}/samples.jsonl"
  : > "${output_dir}/keys"
  : > "${output_dir}/wav.scp"
  for ((source_index=1; source_index<=num_sources; source_index++)); do
    : > "${output_dir}/ref${source_index}.scp"
  done
  mix_wavs=("${mix_dir}"/*.wav)
  for mix_wav in "${mix_wavs[@]}"; do
    filename=${mix_wav##*/}
    key=${filename%.wav}
    read_speakers "${key}"
    {
      printf '{"key":'
      json_string "${key}"
      printf ',"spk":['
      for ((source_index=0; source_index<num_sources; source_index++)); do
        if ((source_index > 0)); then printf ','; fi
        json_string "${spk_ids[source_index]}"
      done
      printf '],"mix":{"default":['
      json_string "${mix_wav}"
      printf ']},"src":{'
      for ((source_index=1; source_index<=num_sources; source_index++)); do
        if ((source_index > 1)); then printf ','; fi
        json_string "${spk_ids[source_index-1]}"
        printf ':['
        json_string "${mix_data_path}/${split}/s${source_index}/${filename}"
        printf ']'
      done
      printf '}}\n'
    } >> "${output_dir}/samples.jsonl"
    printf '%s\n' "${key}" >> "${output_dir}/keys"
    printf '%s %s\n' "${key}" "${mix_wav}" >> "${output_dir}/wav.scp"
    for ((source_index=1; source_index<=num_sources; source_index++)); do
      printf '%s %s\n' "${key}" "${mix_data_path}/${split}/s${source_index}/${filename}" >> "${output_dir}/ref${source_index}.scp"
    done
  done
  # WeSep raw.list contains JSON records, not paths to manifest files.
  cp "${output_dir}/samples.jsonl" "${output_dir}/raw.list"
  echo "Prepared ${split}: ${#mix_wavs[@]} mixtures, ${num_sources} sources -> ${output_dir}"
done
