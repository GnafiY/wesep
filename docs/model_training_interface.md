# Model Training Interface

> Last updated: 2026-09-21
> Language: English | [中文](model_training_interface.zh-CN.md)

This document records the current interface between collate, models, losses,
the executor, and checkpoints.

## Model Loading

`wesep.models.get_model` uses a lazy registry. Official names resolve to a
`module:class` target, and only the selected module is imported.

```python
model_class = get_model(configs["model"]["tse_model"])
model = model_class(
    configs["model_args"]["tse_model"],
    defer_pretrained=defer_pretrained,
)
```

The current registry contains speaker-, spatial-, visual-, textual-, and
speaker-plus-spatial TSE models. The registry also contains internal
experimental implementations; presence in the registry does not by itself
mean a model is advertised as an official release model.

## Model Input

The executor passes the complete named batch to the model:

```python
outputs = model(batch)
```

Tensor values are moved to the selected device without changing dtype.
Strings and other Python metadata remain on the CPU.

Standard fields are:

```text
wav_mix:       FloatTensor [B, C, T]
wav_target:    FloatTensor [B, 1, T]
audio_aux:     optional cue tensor
spatial_aux:   optional cue tensor
visual_aux:    optional cue tensor
textual_aux:   optional cue tensor
*_present:     optional BoolTensor [B]
speaker_label: optional LongTensor [B]
key, spk:      metadata lists
```

Each model reads only the fields it needs.

## Model Output

Models return a named dictionary. The primary waveform contract is:

```python
return {
    "speech": speech,  # FloatTensor [B, S, T]
}
```

Target speaker extraction normally uses `S = 1`. Models with auxiliary
objectives may add outputs such as `speaker_logits`.

## Loss Routing

`LossManager` routes named model outputs to named batch targets:

```yaml
loss:
  - type: SISDR
    output: speech
    target: wav_target
    weight: 1.0
    stages: [train, val]
```

Implemented loss names are `L1`, `L2`, `CE`, `STFT`,
`MultiResolutionSTFT`, `SISDR`, `SISNR`, and `SNR`. At least one loss
must be enabled for each of the `train` and `val` stages.

## Executor Responsibilities

The executor:

1. moves tensors to the device;
2. calls `model(batch)`;
3. evaluates the configured losses;
4. performs backward, gradient clipping, optimizer updates, and scheduler
   hooks;
5. reports rank-aware training and validation metrics.

The executor does not interpret cue modality semantics.

## Training Launcher

The current official recipes launch `wesep/bin/train.py` with `torchrun`.
Training currently assumes CUDA, NCCL, and the distributed environment
variables provided by `torchrun`, even for a one-GPU run.

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

Recipe scripts should be preferred because they resolve these paths and values.

## Initialization and Resume

- `model_init.tse_model` loads model weights for fine-tuning but does not
  restore optimizer, scheduler, scaler, epoch, or checkpoint metadata.
- `checkpoint` resumes the training state and starts from the next checkpoint
  epoch.
- External frontend weights are deferred when a complete TSE checkpoint will
  be loaded.
- Training writes the resolved config to `exp_dir/config.yaml` and model
  checkpoints to `exp_dir/models/`.
- `average_model.py` writes an averaged checkpoint containing a `models`
  list compatible with `load_pretrained_model`.

## Current Boundaries

- CPU-only training is not an official path.
- Waveform length masks and masked SI-SDR are not implemented.
- Blind separation collation and PIT are not implemented.
- The current JIT exporter and legacy C++ runtime do not implement the named
  batch interface.
- CLI and released pretrained model packages are still under validation.
