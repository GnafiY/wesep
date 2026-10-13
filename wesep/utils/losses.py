import auraloss
import torch
import torch.nn as nn

from wesep.utils.bss_losses import PITSISDRLoss

LOSS_REGISTRY = {
    "L1": nn.L1Loss,
    "L2": nn.MSELoss,
    "CE": nn.CrossEntropyLoss,
    "STFT": auraloss.freq.STFTLoss,
    "MultiResolutionSTFT": auraloss.freq.MultiResolutionSTFTLoss,
    "SISDR": auraloss.time.SISDRLoss,
    "SISNR": auraloss.time.SISDRLoss,
    "SNR": auraloss.time.SNRLoss,
    "PIT_SISDR": PITSISDRLoss,
    "PIT_SISNR": PITSISDRLoss,
}


class LossManager(nn.Module):
    """Build and apply configured losses to named model and batch fields."""

    def __init__(self, loss_conf, use_valid_length_for_sisnr_loss=False):
        super().__init__()
        if isinstance(loss_conf, dict):
            loss_conf = [loss_conf]
        if not isinstance(loss_conf, list):
            raise TypeError("loss configuration must be a dict or list")
        if not loss_conf:
            raise ValueError("loss configuration must not be empty")

        self.losses = nn.ModuleList()
        self.specs = []
        self.use_valid_length_for_sisnr_loss = bool(
            use_valid_length_for_sisnr_loss)

        # Build every configured loss once and retain its routing metadata.
        for conf in loss_conf:
            loss_type = conf["type"]
            if loss_type not in LOSS_REGISTRY:
                raise KeyError(f"Unknown loss type: {loss_type}")

            stages = conf.get("stages", ["train"])
            if isinstance(stages, str):
                stages = [stages]
            unknown_stages = set(stages) - {"train", "val"}
            if unknown_stages:
                raise ValueError(
                    f"Unknown loss stages: {sorted(unknown_stages)}")

            loss_fn = LOSS_REGISTRY[loss_type](**conf.get("args", {}))
            self.losses.append(loss_fn)
            self.specs.append({
                "type": loss_type,
                "output": conf["output"],
                "target": conf["target"],
                "weight": float(conf.get("weight", 1.0)),
                "stages": tuple(stages),
            })

        for stage in ("train", "val"):
            if not self.has_stage(stage):
                raise ValueError(
                    f"at least one loss must include the {stage!r} stage")

    def has_stage(self, stage):
        return any(stage in spec["stages"] for spec in self.specs)

    @staticmethod
    def _mean_loss(loss_fn, estimate, target):
        value = loss_fn(estimate, target)
        return value.mean() if value.ndim > 0 else value

    def _waveform_loss(self, loss_fn, spec, estimate, target, valid_lengths):
        """Exclude right-padding from configured SI and PIT waveform losses."""
        if (not self.use_valid_length_for_sisnr_loss
                or spec["type"] not in {"SISDR", "SISNR", "PIT_SISDR", "PIT_SISNR"}
                or valid_lengths is None):
            return self._mean_loss(loss_fn, estimate, target)

        if estimate.ndim < 2 or target.ndim < 2:
            raise ValueError(
                "Valid-length SI loss requires batched waveform tensors")
        if estimate.shape[0] != target.shape[0]:
            raise ValueError("Estimate and target batch sizes must match")
        valid_lengths = torch.as_tensor(
            valid_lengths, device=estimate.device, dtype=torch.long)
        if valid_lengths.ndim != 1 or valid_lengths.numel() != estimate.shape[0]:
            raise ValueError(
                "valid_lengths must contain one length per waveform target")
        if (torch.any(valid_lengths <= 0)
                or torch.any(valid_lengths > estimate.shape[-1])):
            raise ValueError(
                f"Invalid valid_lengths for waveform length "
                f"{estimate.shape[-1]}")

        total = estimate.new_zeros(())
        for length in valid_lengths.unique(sorted=True).detach().cpu().tolist():
            selected = valid_lengths == length
            weight = selected.sum().to(estimate.dtype) / estimate.shape[0]
            total = total + weight * self._mean_loss(
                loss_fn,
                estimate[selected, ..., :length],
                target[selected, ..., :length],
            )
        return total

    def forward(self, outputs, batch, stage):
        """Return the weighted loss sum enabled for the requested stage."""
        total_loss = None
        for loss_fn, spec in zip(self.losses, self.specs):
            if stage not in spec["stages"]:
                continue
            if spec["output"] not in outputs:
                raise KeyError(
                    f"Model output missing loss field: {spec['output']}")
            if spec["target"] not in batch:
                raise KeyError(f"Batch missing loss target: {spec['target']}")

            value = self._waveform_loss(
                loss_fn,
                spec,
                outputs[spec["output"]],
                batch[spec["target"]],
                batch.get("valid_lengths"),
            )
            value = spec["weight"] * value
            total_loss = value if total_loss is None else total_loss + value

        if total_loss is None:
            raise RuntimeError(f"No loss is enabled for stage: {stage}")
        return total_loss
