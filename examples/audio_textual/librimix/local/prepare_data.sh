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

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

if [ -z "${mix_data_path}" ] || [ ! -d "${mix_data_path}" ]; then
  echo "--mix-data-path must name an existing LibriMix wav16k/min directory." >&2
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

audio_recipe=${WESEP_ROOT}/examples/audio/librimix
textual_recipe=${WESEP_ROOT}/examples/textual/librimix_keywords
phoneme_map=${resource_dir}/phoneme2int.txt
lexicon=${resource_dir}/word2lexicon.txt
transcript_jsonl=${resource_dir}/transcript_nemo/transcript_nemo.jsonl
alignment_dir=${resource_dir}/alignments
for path in "${phoneme_map}" "${lexicon}" "${transcript_jsonl}"; do
  if [ ! -f "${path}" ]; then
    echo "KCE resource missing: ${path}; run Stage 0 first." >&2
    exit 1
  fi
done
if ! find "${alignment_dir}" -type f -name '*.alignment.txt' -print -quit 2>/dev/null | grep -q .; then
  echo "Extracted LibriSpeech alignments missing under ${alignment_dir}; run Stage 0 first." >&2
  exit 1
fi

for split in train-100 dev test; do
  split_dir=${data}/${noise_type}/${split}
  mkdir -p "${split_dir}/cues"
  python "${audio_recipe}/local/scan_librimix.py" \
    "${mix_data_path}/${split}/mix_${noise_type}" \
    --outfile "${split_dir}/samples.jsonl"
  ln -sf samples.jsonl "${split_dir}/raw.list"
done

python "${audio_recipe}/local/build_audio_cues.py" \
  --samples_jsonl "${data}/${noise_type}/train-100/samples.jsonl" \
  --outfile "${data}/${noise_type}/train-100/cues/audio.json"

for split in dev test; do
  cue_dir=${data}/${noise_type}/${split}/cues
  python "${audio_recipe}/local/build_audio_cues.py" \
    --samples_jsonl "${data}/${noise_type}/${split}/samples.jsonl" \
    --outfile "${cue_dir}/audio.json"
  url="https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/main/egs/libri2mix/data/wav8k/min/${split}/map_mixture2enrollment"
  wget -O "${cue_dir}/mixture2enrollment" "${url}"
  python "${audio_recipe}/local/build_fixed_enroll_from_BUT.py" \
    --mixture2enrollment "${cue_dir}/mixture2enrollment" \
    --speech_json "${cue_dir}/audio.json" \
    --outfile "${cue_dir}/fixed_enroll.json"
done

python "${textual_recipe}/local/dae_kce/build_transcript_cues.py" \
  --samples "${data}/${noise_type}/train-100/samples.jsonl" \
  --subset train-100 \
  --transcript-nemo-jsonl "${transcript_jsonl}" \
  --phoneme-map "${phoneme_map}" \
  --lexicon "${lexicon}" \
  --output "${data}/${noise_type}/train-100/cues/textual.json"
python "${textual_recipe}/local/dae_kce/build_transcript_cues.py" \
  --samples "${data}/${noise_type}/dev/samples.jsonl" \
  --subset dev \
  --transcript-nemo-jsonl "${transcript_jsonl}" \
  --phoneme-map "${phoneme_map}" \
  --lexicon "${lexicon}" \
  --fixed-words "${dev_keyword_count}" \
  --seed 42 \
  --output "${data}/${noise_type}/dev/cues/textual.json"

for count in 1 2 3 4; do
  official_cues=${resource_dir}/testset/kw-${count}_seed-42.jsonl
  if [ ! -f "${official_cues}" ]; then
    echo "Official KCE test cues missing: ${official_cues}" >&2
    exit 1
  fi
  python "${textual_recipe}/local/dae_kce/convert_official_cues.py" \
    --samples "${data}/${noise_type}/test/samples.jsonl" \
    --official-cues "${official_cues}" \
    --output "${data}/${noise_type}/test/cues/textual_kw${count}.json"
done
ln -sf "textual_kw${test_keyword_count}.json" \
  "${data}/${noise_type}/test/cues/textual.json"

cat > "${data}/${noise_type}/train-100/cues.yaml" <<EOF
cues:
  audio:
    type: wav
    format: waveform
    guaranteed: true
    scope: speaker
    policy:
      type: random
      key: spk_id
      resource: ${data}/${noise_type}/train-100/cues/audio.json
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
      chunk_aware:
        alignment_path: ${alignment_dir}
        word2lexicon_path: ${lexicon}
        tolerance: 0.0001
        no_word_policy: discard
    collate:
      fill_value: 0
      emit_present: false
EOF

for split in dev test; do
  cat > "${data}/${noise_type}/${split}/cues.yaml" <<EOF
cues:
  audio:
    type: wav
    format: waveform
    guaranteed: true
    scope: speaker
    policy:
      type: fixed
      key: mix_spk_id
      resource: ${data}/${noise_type}/${split}/cues/fixed_enroll.json
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
