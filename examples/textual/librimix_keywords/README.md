# LibriMix Keyword-Cued TSE

> Last updated: 2026-09-21

This recipe prepares the keyword phoneme cues used by DAE-TSE and trains a
text-aware BSRNN on Libri2Mix-100 mix-clean. It follows the public DAE-TSE
resources at https://github.com/GnafiY/DAE-TSE.

## Data preparation

Stage 0 downloads the frozen KCE checkpoint, its model configuration, the
checkpoint-compatible phoneme map and lexicon, and the official one- to
four-keyword Libri2Mix test manifests.

Stage 1 expects an existing `Libri2Mix/wav16k/min` tree and the corresponding
LibriSpeech transcripts. It generates standard WeSep `samples.jsonl` files,
word-aligned phoneme cue indexes, and explicit `cues.yaml` files. Training cues
retain the complete transcript so the DAE processor can select two to six
contiguous words. Development and test cues use fixed keyword positions.

Install the lightweight text frontend before running Stage 1:

```bash
pip install g2p_en
```

```bash
./run.sh --stage 0 --stop_stage 1 \
  --librimix_root /path/to/Libri2Mix \
  --librispeech_root /path/to/LibriSpeech
```

Set `--test_keyword_count` to `1`, `2`, `3`, or `4` to choose the corresponding
official DAE-TSE test condition. The generated `test/cues/textual_kwN.json`
files remain available for evaluating every condition without rebuilding data.

The model receives padded phoneme IDs rather than raw strings. Both this recipe
and direct text inference use `DAEPhonemeTokenizer` from
`wesep.modules.textual.kce`, ensuring identical normalization and phoneme IDs.

The official configuration uses `batch_size: 1` and `whole_utt: true`, matching
the full-context KCE training setup.

## Stages

Stages 0–6 download the text resources, prepare audio and keyword cue indexes,
optionally create shards, train, average checkpoints, infer, and score. Data
preparation and training are the validated v0.1 path; inference and scoring
remain under release-wide validation.

## Citation

```bibtex
@inproceedings{ijcai2026-haoyuli-daetse,
  title     = {Detect, Attend and Extract: Keyword Guided Target Speaker Extraction},
  author    = {Li, Haoyu and Xi, Yu and Jiang, Yidi and Wang, Shuai and Knill, Kate and Gales, Mark and Li, Haizhou and Yu, Kai},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on
               Artificial Intelligence, {IJCAI-26}},
  pages     = {5784--5792},
  year      = {2026},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2026/644},
  url       = {https://doi.org/10.24963/ijcai.2026/644},
}
```
