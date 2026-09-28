# Audio-textual Libri2Mix recipe

This recipe trains a mono BSRNN target speaker extraction model with two cues:

- a same-speaker enrollment utterance represented by a TFMap;
- a phoneme keyword sequence encoded by frozen pretrained DAE KCE, then fused
  into each BSRNN block through its trainable adapter.

Data preparation reuses the standard LibriMix scanner and audio enrollment
builder, plus the textual recipe's transcript, lexicon, phoneme, and alignment
utilities. Stage 0 fetches public KCE/text/alignment resources; stage 1 builds
train/dev/test indexes and cue manifests. Set `librimix_root` to the directory
containing `wav16k/min/{train-100,dev,test}` before running stages 0–1.

```bash
bash run.sh --stage 0 --stop_stage 1 \
  --librimix_root /path/to/Libri2Mix \
  --gpus '[0,1]'
```

Stages 2–6 create shards, train, average checkpoints, infer, and score. With
the default chunk configuration, train samples use alignment-aware keyword
selection and validation remains full-utterance.
