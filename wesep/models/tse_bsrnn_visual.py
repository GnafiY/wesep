# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from wesep.modules.separator.bsrnn import BSRNN
from wesep.modules.visual.visual_frontend import VisualFrontend
from wesep.modules.common.deep_update import deep_update


class TSE_BSRNN_VISUAL(nn.Module):
    """Extract reference-channel speech using a visual target cue."""

    def __init__(self, config, defer_pretrained=False):
        super().__init__()

        separator_config = {
            "sr": 16000,
            "win": 512,
            "stride": 128,
            "feature_dim": 128,
            "num_repeat": 6,
            "causal": False,
            "nspk": 1,
            "channels": 1,
            "reference_channel": 0,
        }
        separator_config.update(config.get("separator", {}))
        separator_config["spec_dim"] = 2 * separator_config["channels"]
        self.sep_model = BSRNN(**separator_config)

        visual_configs = {
            "features": {
                "muse_visual": {
                    "mix_dim": separator_config["feature_dim"],
                    "repeat": separator_config["num_repeat"],
                }
            }
        }
        self.visual_configs = deep_update(visual_configs, config["visual"])
        self.visual_ft = VisualFrontend(
            self.visual_configs,
            defer_pretrained=defer_pretrained,
        )
        self.multi_fuse = self.visual_ft.muse_visual.multi_fuse

    def forward(self, batch):
        """Use `wav_mix` [B,C,T] and temporal `visual_aux` cues."""
        mix = batch["wav_mix"]
        visual_aux = batch["visual_aux"]
        if mix.ndim == 2 and self.sep_model.channels == 1:
            mix = mix.unsqueeze(1)
        if mix.ndim != 3 or mix.shape[1] != self.sep_model.channels:
            raise ValueError(
                f"Expected [B, {self.sep_model.channels}, T] mixture, got "
                f"{tuple(mix.shape)}")

        # S1. Convert every input channel into a complex spectrum.
        mix_spec = self.sep_model.stft(mix)[-1]  # [B, C, F, T]
        spectral_repr = torch.cat(
            [mix_spec.real, mix_spec.imag],
            dim=1,
        )  # [B, 2*C, F, T]

        # S2. Split the input and reference spectra into frequency bands.
        subband_spec = self.sep_model.band_split(spectral_repr,
                                                 )  # N_b * [B, 2*C, BW, T]
        reference_spec = mix_spec[:, self.sep_model.reference_channel,
                                  ]  # [B, F, T]
        subband_mix_spec = self.sep_model.band_split(reference_spec,
                                                     )  # N_b * [B, BW, T]

        # S3. Normalize and project each band into the separator dimension.
        subband_feature = self.sep_model.subband_norm(subband_spec,
                                                      )  # [B, N_b, E, T]

        # V1. Compute and align the shared Muse-like visual representation.
        if hasattr(self.visual_ft, "muse_visual"):
            visual_feature = self.visual_ft.muse_visual.compute(
                visual_aux,
                mix=subband_feature,
            )  # [B, D_v, T]
            visual_feature = visual_feature.unsqueeze(1)  # [B, 1, D_v, T]

        # S4. Inject once before the first block, or independently per block.
        sep_output = subband_feature
        for index in range(self.sep_model.separator.num_blocks):
            if (hasattr(self.visual_ft, "muse_visual")
                    and (self.multi_fuse or index == 0)):
                sep_output = self.visual_ft.muse_visual.post(
                    sep_output,
                    visual_feature,
                    stage=index if self.multi_fuse else 0,
                )
            sep_output = self.sep_model.separator.forward_block(
                sep_output,
                index,
            )  # [B, N_b, E, T]

        # S5. Apply complex masks to the configured reference channel.
        est_spec_ri = self.sep_model.band_masker(
            sep_output,
            subband_mix_spec,
        )  # [B, 2, S, F, T]
        est_spec = torch.complex(
            est_spec_ri[:, 0],
            est_spec_ri[:, 1],
        )  # [B, S, F, T]

        # S6. Convert the estimated reference spectrum back to waveform.
        speech = self.sep_model.istft(
            est_spec,
            length=mix.shape[-1],
        )  # [B, S, L]
        return {"speech": speech}


def check_causal(model):
    fs = 16000
    fps = 25
    input = torch.randn(1, 1, fs * 8).clamp_(-1, 1)
    enroll = torch.randn(1, 32, fps * 8).clamp_(-1, 1)
    model = model.eval()
    with torch.no_grad():
        out1 = model({
            "wav_mix": input,
            "visual_aux": enroll,
        })["speech"]
        for i in range(1, 4, 1):
            inputs2 = input.clone()
            t = i * fs
            inputs2[..., t:] = 1 + torch.rand_like(inputs2[..., t:])
            enroll2 = enroll.clone()
            t = i * fps
            enroll2[..., t:] = 1 + torch.rand_like(enroll2[..., t:])
            out2 = model({
                "wav_mix": inputs2,
                "visual_aux": enroll2,
            })["speech"]
            print((((out1[0] - out2[0]).abs() > 1e-8).float().argmax()) / fs)
            print((((inputs2 - input).abs() > 1e-8).float().argmax()) / fs)


if __name__ == "__main__":
    config = {
        "separator": {
            "feature_dim": 32,
            "num_repeat": 2,
            "channels": 1,
            "causal": True,
        },
        "visual": {
            "features": {
                "muse_visual": {
                    "input": "muse_frontend",
                    "vf_pretrained": None,
                    "upsample": True,
                    "fusion": "concat",
                    "multi_fuse": False,
                    "adapter": {
                        "type": "direct",
                        "input_dim": 32,
                        "output_dim": 32,
                    },
                },
            },
        },
    }
    model = TSE_BSRNN_VISUAL(config).eval()
    batch = {
        "wav_mix": torch.randn(2, 1, 16000),
        "visual_aux": torch.randn(2, 32, 25),
    }
    with torch.no_grad():
        output = model(batch)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameters / 1e6:.2f} M")
    print(f"Output: {tuple(output['speech'].shape)}")

    check_causal(model)
