# WeSep 发布路线图

> 最后更新：2026-09-21
> 语言：[English](release_roadmap.md) | 中文

WeSep v0.1 定位为面向学术开发的研究预览版。当前经过验证的范围是由维护中的
recipe 完成数据准备与模型训练。推理工具已经存在，但 CLI 和对外发布的预训练模型仍在
验证，将于近期发布；打包安装与部署属于后续工作。

## v0.1 已完成

- [x] 统一 raw list、shard、sample manifest 与 cue registry 数据路径。
- [x] 维护中的音频 recipe 支持离线与在线混合数据准备。
- [x] speaker、spatial、visual、textual cue 的 dataset 与 collate 路径。
- [x] 使用具名 batch 字段的模块化 separator 与 cue 插入接口。
- [x] 单通道 speaker-cue LibriMix 训练 recipe。
- [x] 多通道 speaker-cue 与 spatial-cue LibriMix 训练 recipe。
- [x] 联合 speaker-plus-spatial LibriMix 训练 recipe。
- [x] VoxCeleb2Mix raw-video 与 LibriMix keyword 训练 recipe。
- [x] 核心数据和训练约定的中英文文档。

## 近期发布

- [ ] 完成各 recipe 推理与评分流程的端到端验证。
- [ ] 验证并发布使用本地 checkpoint 的 CLI。
- [ ] 发布官方预训练 checkpoint，并支持自动下载。
- [ ] 增加 raw/shard 加载、cue 对齐、cue 缺失、在线混合与模型 forward 的
  针对性 smoke test。
- [ ] 发布维护中各 recipe 的可复现实验基线。

## v0.1 之后的计划

- [ ] 更新打包元数据，提供经过测试的 wheel/PyPI 安装方式。
- [ ] 定义并验证 ONNX 或 TorchScript 部署接口。
- [ ] 在接口稳定后替换或更新旧 C++ runtime。
- [ ] 仅在数据、训练、推理和评测流程可复现后，扩展多 cue recipe 与 separator
  backbone。

当前详细约定见 [`data_pipeline.zh-CN.md`](data_pipeline.zh-CN.md)、
[`cue_collate_design.zh-CN.md`](cue_collate_design.zh-CN.md) 和
[`model_training_interface.zh-CN.md`](model_training_interface.zh-CN.md)。
