#!/usr/bin/env bash

set -euo pipefail

if [ -z "${WESEP_ROOT:-}" ]; then
  . "$(dirname "${BASH_SOURCE[0]}")/../path.sh"
fi

mix_data_path=
librispeech_root=
resource_dir=exp/kce
data=data
noise_type=clean
dev_keyword_count=4
test_keyword_count=4

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

if [ -z "${mix_data_path}" ] || [ -z "${librispeech_root}" ]; then
  echo "--mix-data-path and --librispeech-root are required." >&2
  exit 1
fi
if [ ! -d "${mix_data_path}" ] || [ ! -d "${librispeech_root}" ]; then
  echo "Libri2Mix or LibriSpeech data root does not exist." >&2
  exit 1
fi
if [[ ! "${test_keyword_count}" =~ ^[1-4]$ ]]; then
  echo "--test-keyword-count must be one of 1, 2, 3, or 4." >&2
  exit 1
fi
if [[ ! "${dev_keyword_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--dev-keyword-count must be a positive integer." >&2
  exit 1
fi

phoneme_map=${resource_dir}/phoneme2int.txt
lexicon=${resource_dir}/word2lexicon.txt
for path in "${phoneme_map}" "${lexicon}"; do
  if [ ! -f "${path}" ]; then
    echo "DAE text resource not found: ${path}; run Stage 0 first." >&2
    exit 1
  fi
done

# Build standard WeSep audio indexes directly from Libri2Mix.
for split in train-100 dev test; do
  output_dir=${data}/${noise_type}/${split}
  mkdir -p "${output_dir}/cues"
  python "${WESEP_ROOT}/examples/audio/librimix/local/scan_librimix.py" \
    "${mix_data_path}/${split}/mix_${noise_type}" \
    --outfile "${output_dir}/samples.jsonl"
  ln -sf samples.jsonl "${output_dir}/raw.list"
done

# Preserve complete word-aligned transcripts for random training selection.
python local/dae_kce/build_transcript_cues.py \
  --samples "${data}/${noise_type}/train-100/samples.jsonl" \
  --librispeech-root "${librispeech_root}" \
  --phoneme-map "${phoneme_map}" \
  --lexicon "${lexicon}" \
  --output "${data}/${noise_type}/train-100/cues/textual.json"

# Validation uses one deterministic keyword selection across all epochs.
python local/dae_kce/build_transcript_cues.py \
  --samples "${data}/${noise_type}/dev/samples.jsonl" \
  --librispeech-root "${librispeech_root}" \
  --phoneme-map "${phoneme_map}" \
  --lexicon "${lexicon}" \
  --fixed-words "${dev_keyword_count}" \
  --seed 42 \
  --output "${data}/${noise_type}/dev/cues/textual.json"

# Convert all official test conditions and select one as the recipe default.
for count in 1 2 3 4; do
  official_cues=${resource_dir}/testset/kw-${count}_seed-42.jsonl
  if [ ! -f "${official_cues}" ]; then
    echo "Official DAE test cues not found: ${official_cues}" >&2
    exit 1
  fi
  python local/dae_kce/convert_official_cues.py \
    --samples "${data}/${noise_type}/test/samples.jsonl" \
    --official-cues "${official_cues}" \
    --output "${data}/${noise_type}/test/cues/textual_kw${count}.json"
done
ln -sf "textual_kw${test_keyword_count}.json" \
  "${data}/${noise_type}/test/cues/textual.json"

# Write the DAE-specific representation and selection policy explicitly.
cat > "${data}/${noise_type}/train-100/cues.yaml" <<EOF
cues:
  textual:
    type: json
    format: dae_keyword_phoneme
    guaranteed: true
    scope: speaker
    policy:
      type: fixed
      key: mix_spk_id
      resource: ${data}/${noise_type}/train-100/cues/textual.json
    selection:
      type: random
      word_range: [2, 6]
    collate:
      fill_value: 0
      emit_present: false
EOF

for split in dev test; do
  cat > "${data}/${noise_type}/${split}/cues.yaml" <<EOF
cues:
  textual:
    type: json
    format: dae_keyword_phoneme
    guaranteed: true
    scope: speaker
    policy:
      type: fixed
      key: mix_spk_id
      resource: ${data}/${noise_type}/${split}/cues/textual.json
    selection:
      type: fixed
    collate:
      fill_value: 0
      emit_present: false
EOF
done
