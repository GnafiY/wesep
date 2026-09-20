"""Textual cue frontend for keyword-conditioned target extraction."""

import torch.nn as nn

from wesep.modules.common.deep_update import deep_update
from wesep.modules.fusion.speech import SpeakerFuseLayer
from wesep.modules.textual.kce import KCE


class DAEKCEFeature(nn.Module):
    """Expose KCE computation and separator fusion as one textual feature."""

    def __init__(self, config, defer_pretrained=False):
        super().__init__()
        self.kce = KCE(config, defer_pretrained=defer_pretrained)
        self.multi_fuse = config["multi_fuse"]
        num_fusions = config["num_stages"] if self.multi_fuse else 1
        self.fusions = nn.ModuleList([
            SpeakerFuseLayer(
                embed_dim=self.kce.output_dim,
                feat_dim=config["mix_dim"],
                fuse_type=config["fusion"],
            ) for _ in range(num_fusions)
        ])

    def compute(self, mix, textual_aux):
        """Compute one adapted KCE cue [B,D]."""
        return self.kce(mix, textual_aux.long())

    def post(self, mix, cue, stage):
        """Fuse the cue before the selected separator block."""
        return self.fusions[stage](mix, cue.unsqueeze(1).unsqueeze(-1))


class TextualFrontend(nn.Module):
    """Build enabled textual features behind a small common interface."""

    def __init__(self, config, defer_pretrained=False):
        super().__init__()
        default_config = {
            "features": {
                "dae_kce": {
                    "enabled": False,
                    "padding_id": 0,
                    "bos_id": 71,
                    "eos_id": 72,
                    "adapter": {
                        "output_dim": 128,
                        "num_layers": 1,
                    },
                    "fusion": "FiLM",
                    "multi_fuse": True,
                    "mix_dim": 128,
                    "num_stages": 6,
                    "pretrained": None,
                }
            },
        }
        self.config = deep_update(default_config, config)
        feature = self.config["features"]["dae_kce"]
        if not feature["enabled"]:
            raise ValueError("TextualFrontend requires dae_kce to be enabled")
        self.dae_kce = DAEKCEFeature(feature,
                                     defer_pretrained=defer_pretrained)
