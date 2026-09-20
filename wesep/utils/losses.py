import auraloss
import torch.nn as nn

LOSS_REGISTRY = {
    "L1": nn.L1Loss,
    "L2": nn.MSELoss,
    "CE": nn.CrossEntropyLoss,
    "STFT": auraloss.freq.STFTLoss,
    "MultiResolutionSTFT": auraloss.freq.MultiResolutionSTFTLoss,
    "SISDR": auraloss.time.SISDRLoss,
    "SISNR": auraloss.time.SISDRLoss,
    "SNR": auraloss.time.SNRLoss,
}


class LossManager(nn.Module):
    """Build and apply configured losses to named model and batch fields."""

    def __init__(self, loss_conf):
        super().__init__()
        if isinstance(loss_conf, dict):
            loss_conf = [loss_conf]
        if not isinstance(loss_conf, list):
            raise TypeError("loss configuration must be a dict or list")
        if not loss_conf:
            raise ValueError("loss configuration must not be empty")

        self.losses = nn.ModuleList()
        self.specs = []

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
                    f"at least one loss must include the '{stage}' stage")

    def has_stage(self, stage):
        return any(stage in spec["stages"] for spec in self.specs)

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

            value = loss_fn(outputs[spec["output"]], batch[spec["target"]])
            if value.ndim > 0:
                value = value.mean()
            value = spec["weight"] * value
            total_loss = value if total_loss is None else total_loss + value

        if total_loss is None:
            raise RuntimeError(f"No loss is enabled for stage: {stage}")
        return total_loss
