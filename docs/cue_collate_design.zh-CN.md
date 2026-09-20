# Cue 与 Collate 接口契约

> 最后更新：2026-09-21
> 语言：[English](cue_collate_design.md) | 中文

本文档定义当前实现中 cue metadata、processor、collate 和模型 frontend
之间的职责边界。

## 职责划分

- `cues.yaml` 描述某个数据 split 实际包含什么 cue，以及如何查找。
- 各模态 processor 读取资源，并转换为布局明确的 tensor。
- `build_collect_keys` 选择当前实验需要的字段。
- `tse_collate_fn` 展开 speaker 轴、对齐最后一维、构造 optional fallback
  并 stack tensor。
- 模型 frontend 解释 cue 语义，并决定缺失 cue 如何参与计算。

Collate 不解释 waveform、video、direction 或 text 的真实语义。

## Split 级 Cue Schema

```yaml
cues:
  spatial:
    type: npy
    format: fields
    scope: speaker
    guaranteed: true
    fields: [azimuth, elevation]
    policy:
      type: fixed
      key: mix_spk_id
      resource: data/train/cues/spatial.json
    collate:
      default_shape: [2]
      fill_value: -999.0
```

字段含义：

- `type`：物理 payload 或 reader 类型。
- `format`：reader 输出的语义表示。
- `scope`：当前只支持 `speaker`。
- `guaranteed`：该 split 是否保证每个目标都存在 cue。
- `policy.type`：对应模态支持的 `random` 或 `fixed`。
- `policy.key`：`spk_id` 或 `mix_spk_id`。
- `policy.resource`：JSON 查找表。
- `collate.default_shape`：整个 batch 都缺失 cue 时的 fallback shape。
- `collate.fill_value`：缺失 cue 和 constant padding 使用的值。

## 已实现 Cue 格式

`wesep/dataset/cues.py` 的注册表接受：

| 模态 | Type | Format | Sample 字段 |
| --- | --- | --- | --- |
| Audio | `wav` | `waveform` | `audio_spk{i}` |
| Audio | `npy` | `spk_embedding` | `audio_spk{i}` |
| Visual | `mp4` | `raw_video` | `visual_spk{i}` |
| Visual | `npy` | `muse_frontend` | `visual_spk{i}` |
| Spatial | `npy` | `fields` | `spatial_spk{i}` |
| Textual | `json` | `dae_keyword_phoneme` | `textual_spk{i}` |

预计算 `muse_frontend` 仍是实验兼容路径，不作为正式 WeSep 模型宣传。

## 实验级选择

模型实验声明自己消费哪些数据 cue：

```yaml
dataset_args:
  cues:
    audio:
      use: true
      required: true
    spatial:
      use: true
      required: false
      default_shape: [2]
```

`use` 启用 processor 和 collate 字段；`required` 表示每个展开后的 target
是否必须包含该 cue。实验要求 `required: true` 时，split metadata 不能写成
`guaranteed: false`。

## Collect Spec 合并优先级

```text
BASE_COLLECT_KEYS
  < cues.yaml cue.collate
  < dataset_args.cues.<modality>
  < dataset_args.collate.<field>
```

最后一层用于显式字段覆盖，例如验证集 waveform alignment。

## Speaker 轴展开

包含 `N` 个目标说话人的 mixture 会展开为 `N` 条 TSE batch item。

- `wav_mix` 等 mix-axis tensor 为每个目标复制。
- speaker-axis tensor 读取 `wav_spk{i}` 或 `{modality}_spk{i}`。
- `key` 等 metadata 复制，`spk` 选择当前目标。
- 输出目标字段统一命名为 `wav_target`。

## 对齐与缺失 Cue

只有 collect spec 声明 `align` 时，tensor 最后一维才被当作时间轴。

| 字段 | 默认对齐 | Padding |
| --- | --- | --- |
| `wav_mix` | max | constant zero |
| `wav_target` | max | constant zero |
| `audio_aux` | min | crop |
| `spatial_aux` | max | edge |
| `visual_aux` | max | edge |
| `textual_aux` | max | constant zero |

缺失 optional cue 优先复用同 batch 真实 cue 的 shape 和 dtype。整个 batch
都缺失时必须提供 `default_shape`。启用后会输出
`audio_aux_present`、`spatial_aux_present`、`visual_aux_present` 和
`textual_aux_present` 布尔 tensor。

Fallback 值只用于保持结构。当数值 fallback 可能被误当成真实 cue 时，模型
frontend 必须使用 presence mask。

## 时间同步

- Waveform cue 由 reader 重采样。
- Raw video 先与 source audio 的完整时长对齐，再按 chunk metadata 截取。
- 预计算 visual 和动态 spatial 特征约定最后一维为时间，可读取 resource
  item 中的 `duration_sec`。
- Textual phoneme ID 是离散序列，使用零 padding。
- Collate 不推断真实 frame rate。

## 当前边界

- Cue scope 只支持 speaker-level。
- 同一 batch 的非时间维必须一致。
- Visual 和 spatial 资源使用 fixed selection。
- Audio 资源支持 fixed 和 random selection。
- Textual 当前实现 DAE keyword-phoneme 格式。
- 只有消费对应 `*_present` mask 的模型才能完整利用缺失 cue 语义。
