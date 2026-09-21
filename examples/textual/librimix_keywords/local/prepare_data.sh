#!/usr/bin/env bash

set -euo pipefail

if [ -z "${WESEP_ROOT:-}" ]; then
  . "$(dirname "${BASH_SOURCE[0]}")/../path.sh"
fi

mix_data_path=
resource_dir=exp/kce
data=data
noise_type=clean
dev_keyword_count=4
test_keyword_count=4
chunk_aware=false

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

if [ -z "${mix_data_path}" ]; then
  echo "--mix-data-path is required." >&2
  exit 1
fi
if [ ! -d "${mix_data_path}" ]; then
  echo "Libri2Mix data root does not exist: ${mix_data_path}" >&2
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
if [[ "${chunk_aware}" != true && "${chunk_aware}" != false ]]; then
  echo "--chunk-aware must be true or false." >&2
  exit 1
fi

phoneme_map=${resource_dir}/phoneme2int.txt
lexicon=${resource_dir}/word2lexicon.txt
transcript_nemo_jsonl=${resource_dir}/transcript_nemo/transcript_nemo.jsonl
for path in "${phoneme_map}" "${lexicon}" "${transcript_nemo_jsonl}"; do
  if [ ! -f "${path}" ]; then
    echo "DAE resource not found: ${path}; run Stage 0 first." >&2
    exit 1
  fi
done

primary_alignment=${resource_dir}/alignments/LibriSpeech-Alignments.zip
fallback_alignment=${resource_dir}/alignments/LibriSpeech-Alignments-Aligned-by-Qwen3-Aligner.zip
if [[ "${chunk_aware}" == true ]]; then
  for path in "${primary_alignment}" "${fallback_alignment}"; do
    if [ ! -s "${path}" ]; then
      echo "Chunk alignment resource not found or empty: ${path}; run Stage 0 first." >&2
      exit 1
    fi
  done
fi

# Build standard WeSep audio indexes directly from Libri2Mix.
for split in train-100 dev test; do
  output_dir=${data}/${noise_type}/${split}
  mkdir -p "${output_dir}/cues"
  python "${WESEP_ROOT}/examples/audio/librimix/local/scan_librimix.py" \
    "${mix_data_path}/${split}/mix_${noise_type}" \
    --outfile "${output_dir}/samples.jsonl"
  ln -sf samples.jsonl "${output_dir}/raw.list"
done

# Preserve complete NeMo ASR transcripts for random training selection.
python local/dae_kce/build_transcript_cues.py \
  --samples "${data}/${noise_type}/train-100/samples.jsonl" \
  --subset train-100 \
  --transcript-nemo-jsonl "${transcript_nemo_jsonl}" \
  --phoneme-map "${phoneme_map}" \
  --lexicon "${lexicon}" \
  --output "${data}/${noise_type}/train-100/cues/textual.json"

# Validation uses one deterministic NeMo keyword selection across all epochs.
python local/dae_kce/build_transcript_cues.py \
  --samples "${data}/${noise_type}/dev/samples.jsonl" \
  --subset dev \
  --transcript-nemo-jsonl "${transcript_nemo_jsonl}" \
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
EOF
if [[ "${chunk_aware}" == true ]]; then
  cat >> "${data}/${noise_type}/train-100/cues.yaml" <<EOF
      chunk_aware:
        enabled: true
        alignment_path: ${primary_alignment}
        fallback_alignment_path: ${fallback_alignment}
        word2lexicon_path: ${lexicon}
        tolerance: 0.0001
        no_word_policy: discard
EOF
fi
cat >> "${data}/${noise_type}/train-100/cues.yaml" <<EOF
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
