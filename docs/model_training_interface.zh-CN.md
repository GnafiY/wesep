# 模型训练接口

> 最后更新：2026-09-21
> 语言：[English](model_training_interface.md) | 中文

本文记录 collate、模型、损失函数、训练执行器与 checkpoint 之间当前实际使用的接口。

## 模型加载

`wesep.models.get_model` 使用延迟加载注册表。正式模型名称会解析到
`module:class`，运行时只导入所选模型对应的模块。

```python
model_class = get_model(configs["model"]["tse_model"])
model = model_class(
    configs["model_args"]["tse_model"],
    defer_pretrained=defer_pretrained,
)
```

当前注册表包含 speaker、spatial、visual、textual 以及
speaker-plus-spatial TSE 模型，也保留少量内部实验实现。出现在注册表中并不等于该模型属于
对外发布的正式模型列表。

## 模型输入

训练执行器会把完整的具名 batch 传给模型：

```python
outputs = model(batch)
```

Tensor 会在不改变 dtype 的情况下移动到指定设备；字符串等 Python 元数据保留在 CPU。
标准字段如下：

```text
wav_mix:       FloatTensor [B, C, T]
wav_target:    FloatTensor [B, 1, T]
audio_aux:     可选 cue tensor
spatial_aux:   可选 cue tensor
visual_aux:    可选 cue tensor
textual_aux:   可选 cue tensor
*_present:     可选 BoolTensor [B]
speaker_label: 可选 LongTensor [B]
key, spk:      元数据列表
```

每个模型只读取自身需要的字段。

## 模型输出

模型返回具名字典，主要波形输出遵循以下约定：

```python
return {
    "speech": speech,  # FloatTensor [B, S, T]
}
```

目标说话人提取通常使用 `S = 1`。带辅助目标的模型可以增加
`speaker_logits` 等输出。

## 损失路由

`LossManager` 根据名称把模型输出映射到 batch 中的目标：

```yaml
loss:
  - type: SISDR
    output: speech
    target: wav_target
    weight: 1.0
    stages: [train, val]
```

已经实现的损失包括 `L1`、`L2`、`CE`、`STFT`、
`MultiResolutionSTFT`、`SISDR`、`SISNR` 和 `SNR`。
`train` 与 `val` 阶段都必须至少启用一项损失。

## 训练执行器职责

训练执行器负责：

1. 把 tensor 移动到设备；
2. 调用 `model(batch)`；
3. 计算配置的损失；
4. 执行反向传播、梯度裁剪、优化器更新与 scheduler hook；
5. 汇报按 rank 处理的训练和验证指标。

训练执行器不解释不同 cue 模态的语义。

## 训练入口

当前正式 recipe 使用 `torchrun` 启动 `wesep/bin/train.py`。现阶段训练依赖
CUDA、NCCL 以及 `torchrun` 提供的分布式环境变量，单卡训练也采用同一入口。

```sh
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  wesep/bin/train.py --config path/to/config.yaml \
  --gpus "[0]" \
  --train_data path/to/train/raw.list \
  --train_cues path/to/train/cues.yaml \
  --train_samples path/to/train/samples.jsonl \
  --val_data path/to/dev/raw.list \
  --val_cues path/to/dev/cues.yaml \
  --val_samples path/to/dev/samples.jsonl
```

建议优先使用 recipe 脚本，由脚本解析这些路径与配置值。

## 初始化与恢复训练

- `model_init.tse_model` 只加载模型权重用于微调，不恢复优化器、scheduler、scaler、
  epoch 或 checkpoint 元数据。
- `checkpoint` 恢复训练状态，并从 checkpoint 的下一 epoch 开始。
- 加载完整 TSE checkpoint 时会延迟加载外部 frontend 权重。
- 训练会把解析后的配置写入 `exp_dir/config.yaml`，模型 checkpoint 写入
  `exp_dir/models/`。
- `average_model.py` 生成包含 `models` 列表的平均 checkpoint，可由
  `load_pretrained_model` 加载。

## 当前边界

- CPU-only 训练不是正式支持路径。
- 尚未实现波形长度 mask 与 masked SI-SDR。
- 尚未实现盲源分离的 collate 与 PIT。
- 当前 JIT 导出器和旧 C++ runtime 尚未支持具名 batch 接口。
- CLI 与官方预训练模型包仍在验证中。
