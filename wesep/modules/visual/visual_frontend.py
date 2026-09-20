import torch.nn as nn

from wesep.modules.common.deep_update import deep_update
from wesep.modules.visual.muse import MuseVisualFeature


class VisualFrontend(nn.Module):

    def __init__(self, config, defer_pretrained=False):
        super().__init__()

        DEFAULT_CONFIG = {
            "features": {
                "muse_visual": {
                    "enabled": True,
                    "input": "raw_video",
                    "vf_pretrained": "./pretrain_networks/visual_frontend.pt",
                    "freeze_frontend": True,
                    "upsample": False,
                    "mix_dim": 128,
                    "fusion": "concat",
                    "multi_fuse": False,
                    "repeat": 1,
                    "adapter": {
                        "type": "direct",
                        "input_dim": 512,
                        "output_dim": 512,
                    },
                }
            }
        }

        self.config = deep_update(DEFAULT_CONFIG, config)
        feats = self.config["features"]

        # Resolve one cue input shared by every enabled visual feature.
        enabled_inputs = {
            name: conf.get("input", "raw_video")
            for name, conf in feats.items() if conf["enabled"]
        }
        input_types = set(enabled_inputs.values())
        if len(input_types) > 1:
            raise ValueError(
                "Enabled visual features require different cue inputs: "
                f"{enabled_inputs}")
        self.input_type = next(iter(input_types), "raw_video")
        if self.input_type not in ("raw_video", "muse_frontend"):
            raise ValueError(
                f"Unsupported Muse visual input: {self.input_type}")

        if feats["muse_visual"]["enabled"]:
            self.muse_visual = MuseVisualFeature(
                feats["muse_visual"],
                defer_pretrained=defer_pretrained,
            )
