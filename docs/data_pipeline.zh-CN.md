# WeSep 数据流水线契约

> 最后更新：2026-09-21
> 语言：[English](data_pipeline.md) | 中文

本文档描述 `wesep.dataset` 当前已经实现的行为，是面向开发者的代码契约，
不包含尚未实现的设想。

## 流水线总览

`Dataset()` 按以下顺序构建 `IterableDataset`：

```text
DataList
  -> 数据源读取（raw JSONL 或 tar shard）
  -> 重采样和训练分块
  -> 可选在线混合与数据增强
  -> 各模态 cue processor
  -> 可选的评估目标展开
  -> 可选说话人标签
  -> DataLoader
  -> build_collect_keys
  -> tse_collate_fn
  -> model batch
```

在线混合和增强只在 `state == "train"` 时运行。验证和测试读取固定混合语音，
并保留完整 utterance。

## 数据源

### Raw 数据

`raw.list` 每一行是一个 JSON 对象：

```json
{
  "key": "mix_key",
  "spk": ["speaker_a", "speaker_b"],
  "mix": {"default": ["/path/to/mix.wav"]},
  "src": {
    "speaker_a": ["/path/to/target_a.wav"],
    "speaker_b": ["/path/to/target_b.wav"]
  }
}
```

`spk` 定义目标顺序。多个 mixture 文件沿通道轴拼接。当前每个 target 使用
第一个 source 文件的第一通道。source reader 严格检查长度和采样率。

### Shard 数据

离线 shard 中同一样本的成员必须连续：

```text
{key}.spk1
{key}.spk2
{key}.{audio_ext}
{key}_spk1.{audio_ext}
{key}_spk2.{audio_ext}
```

Shard 与 raw JSONL 的样本语义一致，只改变存储和 I/O 方式。

### 分布式迭代

数据列表先按 epoch 可复现地 shuffle，再按 distributed rank 和 DataLoader
worker 切分。训练 shard 少于 worker 时，空 worker 会确定性选择一个兜底
shard，因此可能重复部分训练数据，但不会让 worker 失效。

## 音频处理

所有 waveform 使用 `[C, T]` 布局。

- 所有 state 都重采样到 `dataset_args.resample_rate`。
- `whole_utt: false` 时，训练使用 `random_chunk`。
- 短语音按 `chunk_short_policy` 处理，默认在右侧补零。
- `whole_utt: true` 时保留完整语音，可用 `filter_len` 过滤极端长度。
- 验证和测试不进行随机分块或训练增强。

分块位置以 ratio metadata 保存，供时变 visual/spatial cue 截取对应区间。

## 在线混合

在线混合只在训练阶段接受单说话人 raw 数据或 online shard：

```text
单说话人数据
  -> sample_speaker_group
  -> apply_timeline
  -> 可选单通道混响
  -> SNR mixing
  -> 可选 mixture noise
```

当前在线路径只支持单通道。评估必须使用固定离线 mixture。
`source_key_spk{i}` 保存原始 utterance key，使动态分组后仍能查找
`mix_spk_id` cue。

## Cue 配置

实验配置选择要使用的 cue：

```yaml
dataset_args:
  cues:
    audio:
      use: true
      required: true
```

split 级 `cues.yaml` 描述实际资源：

```yaml
cues:
  audio:
    type: wav
    format: waveform
    scope: speaker
    guaranteed: true
    policy:
      type: random
      key: spk_id
      resource: data/train/cues/audio.json
```

当前实现的表示包括：

| Cue | Type / format | Processor 输出 | Policy |
| --- | --- | --- | --- |
| Audio | `wav / waveform` | `[C, T]` | random 或 fixed |
| Audio | `npy / spk_embedding` | `[D]` | random 或 fixed |
| Visual | `mp4 / raw_video` | `[H, W, C, T]` | fixed |
| Visual | `npy / muse_frontend` | `[..., T]` | fixed，实验兼容路径 |
| Spatial | `npy / fields` | `[F]` 或 `[F, T]` | fixed |
| Textual | `json / dae_keyword_phoneme` | `[L]` 音素 ID | fixed |

当前 cue 都使用 `scope: speaker`。查找键为：

- `spk_id`：`sample["spk{i}"]`
- 离线 `mix_spk_id`：`sample["key"] + "::" + sample["spk{i}"]`
- 在线 `mix_spk_id`：`source_key_spk{i} + "::" + sample["spk{i}"]`

资源 JSON 在每个 DataLoader worker 内按需加载并缓存。

## Collate

`build_collect_keys` 合并基础字段、split cue 默认值和实验覆盖项。
`tse_collate_fn` 将每个 mixture 展开为每个目标说话人一条样本：

```text
输入 mixture：B_in
输出 target：B_out = sum(num_speaker)
```

核心 batch 字段：

```text
wav_mix:       FloatTensor [B_out, C_mix, T]
wav_target:    FloatTensor [B_out, 1, T]
spk:           list[str]
key:           list[str]
num_speaker:   list[int]
speaker_label: optional LongTensor [B_out]
audio_aux:     optional cue tensor
spatial_aux:   optional cue tensor
visual_aux:    optional cue tensor
textual_aux:   optional cue tensor
*_present:     optional BoolTensor [B_out]
```

Waveform 和大多数时变 cue 对齐到 batch 最长项；enrollment waveform 对齐到
最短 enrollment。缺失的 optional cue 使用配置的 fallback tensor，并通过
`*_present` 记录原始 cue 是否存在。

## 当前边界

- Raw 和 shard 使用相同的严格 waveform 契约。
- 当前只支持 speaker-level cue scope。
- 在线混响、SNR mixing 和 mixture noise 是单通道操作。
- 同一 batch 的非时间维必须预先一致。
- Collate 只对齐长度，不重采样 cue frame rate。
- 缓冲 tensor 只提供结构，缺失 cue 的语义由模型 frontend 决定。

正式训练前可运行 `tools/test_dataset.py --config <config>` 检查一个模型输入
batch。
