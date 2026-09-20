# WeSep Release Roadmap

> Last updated: 2026-09-21
> Language: English | [中文](release_roadmap.zh-CN.md)

WeSep v0.1 is a research preview for academic development. Its validated
scope is data preparation and model training through the maintained recipes.
Inference utilities exist, but the CLI and released pretrained models are
still being validated and will be published soon. Packaging and deployment
remain planned work.

## Completed for v0.1

- [x] Unified raw-list, shard, sample-manifest, and cue-registry data paths.
- [x] Offline and online mixture preparation for the maintained audio recipes.
- [x] Speaker-, spatial-, visual-, and textual-cue dataset and collate paths.
- [x] Modular separator and cue-insertion interfaces with named batch fields.
- [x] Single-channel speaker-cue LibriMix training recipe.
- [x] Multichannel speaker-cue and spatial-cue LibriMix training recipes.
- [x] Joint speaker-plus-spatial LibriMix training recipe.
- [x] VoxCeleb2Mix raw-video and LibriMix keyword training recipes.
- [x] English and Chinese documentation for the core data and training
  contracts.

## Coming Soon

- [ ] Complete end-to-end validation of recipe inference and scoring.
- [ ] Validate and publish the CLI for local checkpoints.
- [ ] Release official pretrained checkpoints and automatic model download.
- [ ] Add focused smoke tests for raw/shard loading, cue alignment,
  missing-cue batches, online mixing, and model forward passes.
- [ ] Publish reproducible baseline results for the maintained recipes.

## Planned After v0.1

- [ ] Modernize package metadata and provide a tested wheel/PyPI installation.
- [ ] Define and validate an ONNX or TorchScript deployment contract.
- [ ] Replace or update the legacy C++ runtime after that contract is stable.
- [ ] Expand multi-cue recipes and separator backbones only after their data,
  training, inference, and evaluation paths are reproducible.

The detailed current contracts are documented in
[`data_pipeline.md`](data_pipeline.md),
[`cue_collate_design.md`](cue_collate_design.md), and
[`model_training_interface.md`](model_training_interface.md).
