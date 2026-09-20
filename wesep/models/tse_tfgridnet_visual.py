# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.common.deep_update import deep_update
from wesep.modules.separator.tfgridnet import TFGridNet
from wesep.modules.visual.visual_frontend import VisualFrontend


class TSE_TFGRIDNET_VISUAL(nn.Module):
    """Extract target speech with a visual-conditioned TFGridNet."""

    def __init__(self, config, defer_pretrained=False):
        super().__init__()

        # ===== Merge configs =====
        sep_configs = {
            "nspk": 1,
            "n_fft": 256,
            "stride": 128,
            "channels": 1,
            "n_layers": 6,
            "hidden_channels": 192,
            "n_head": 4,
            "approx_qk_dim": 512,
            "emb_dim": 48,
            "emb_ks": 4,
            "emb_hs": 1,
            "eps": 1.0e-5,
            "causal": False,
            "estimation": "spec",
            "reference_channel": 0,
            "attention_context": None,
            "attention_output_norm": "channel",
        }
        sep_configs = {**sep_configs, **config["separator"]}

        visual_configs = {
            "features": {
                "muse_visual": {
                    "mix_dim": sep_configs["emb_dim"],
                    "repeat": sep_configs["n_layers"],
                }
            }
        }
        self.visual_configs = deep_update(visual_configs, config["visual"])

        # ===== Separator and visual frontend loading =====
        self.sep_model = TFGridNet(**sep_configs)
        self.visual_ft = VisualFrontend(
            self.visual_configs,
            defer_pretrained=defer_pretrained,
        )
        self.multi_fuse = self.visual_ft.muse_visual.multi_fuse

    def forward(self, batch):
        """Use ``wav_mix`` [B,C,L] and temporal ``visual_aux`` cues."""
        mix = batch["wav_mix"]
        visual_aux = batch["visual_aux"]
        if mix.ndim == 2 and self.sep_model.channels == 1:
            mix = mix.unsqueeze(1)
        if mix.ndim != 3 or mix.shape[1] != self.sep_model.channels:
            raise ValueError(
                f"Expected [B,{self.sep_model.channels},T] mixture, got "
                f"{tuple(mix.shape)}")
        mix_len = mix.shape[-1]

        # S1. Convert every input channel into a complex spectrum. ClearerVoice
        # AV-TFGridNet leaves mixture standard-deviation normalization disabled.
        mix_spec = self.sep_model.stft(mix)[-1]  # [B,C,F,T]
        reference_spec = mix_spec[:, self.sep_model.reference_channel,
                                  ]  # [B,F,T]
        spectral_repr = torch.cat(
            [mix_spec.real, mix_spec.imag],
            dim=1,
        )  # [B,2*C,F,T]

        # S2. Project RI spectra into the fixed GridNet feature width.
        spectral_repr = spectral_repr.permute(
            0,
            1,
            3,
            2,
        ).contiguous()  # [B,2*C,T,F]
        if self.sep_model.causal:
            spectral_repr = F.pad(spectral_repr, (0, 0, 2, 0))
        sep_output = self.sep_model.input_norm(
            self.sep_model.input_conv(spectral_repr), )  # [B,E,T,F]

        # V1. Visual fusion treats frequency bins as independent feature bands.
        fusion_output = sep_output.permute(
            0,
            3,
            1,
            2,
        ).contiguous()  # [B,F,E,T]
        visual_feature = self.visual_ft.muse_visual.compute(
            visual_aux,
            mix=fusion_output,
        ).unsqueeze(1)  # [B,1,D_v,T]

        # S3. Inject before the first block, or independently before every
        # block as in ClearerVoice AV-TFGridNet.
        for index in range(self.sep_model.separator.num_blocks):
            if self.multi_fuse or index == 0:
                fusion_output = self.visual_ft.muse_visual.post(
                    fusion_output,
                    visual_feature,
                    stage=index if self.multi_fuse else 0,
                )  # [B,F,E,T]
            sep_output = fusion_output.permute(
                0,
                2,
                3,
                1,
            ).contiguous()  # [B,E,T,F]
            sep_output = self.sep_model.separator.forward_block(
                sep_output,
                index,
            )
            fusion_output = sep_output.permute(
                0,
                3,
                1,
                2,
            ).contiguous()  # [B,F,E,T]

        # S4. Estimate RI spectra for every output source.
        estimated_ri = self.sep_model.output_conv(sep_output,
                                                  )  # [B,2*S,T(+2),F]
        if self.sep_model.causal:
            estimated_ri = estimated_ri[..., :sep_output.shape[-2], :]
        batch_size, _, frames, frequencies = estimated_ri.shape
        estimated_ri = estimated_ri.view(
            batch_size,
            self.sep_model.nspk,
            2,
            frames,
            frequencies,
        )  # [B,S,2,T,F]

        # S5. Interpret the RI pair as a spectrum or complex ratio mask.
        if self.sep_model.estimation == "mask":
            estimated_ri = 5.0 * torch.tanh(estimated_ri / 5.0)
            complex_mask = torch.complex(estimated_ri[:, :, 0],
                                         estimated_ri[:, :, 1])
            estimated_spec = (complex_mask *
                              reference_spec.transpose(-2, -1).unsqueeze(1))
        else:
            estimated_spec = torch.complex(estimated_ri[:, :, 0],
                                           estimated_ri[:, :, 1])
        estimated_spec = estimated_spec.transpose(
            -2,
            -1,
        ).contiguous()  # [B,S,F,T]

        # S6. Convert the estimated spectra back to waveforms.
        speech = self.sep_model.istft(
            estimated_spec,
            length=mix_len,
        )  # [B,S,L]
        return {"speech": speech}


if __name__ == "__main__":
    config = {
        "separator": {
            "n_fft": 128,
            "stride": 64,
            "n_layers": 2,
            "hidden_channels": 32,
            "emb_dim": 16,
            "approx_qk_dim": 256,
        },
        "visual": {
            "features": {
                "muse_visual": {
                    "input": "muse_frontend",
                    "vf_pretrained": None,
                    "upsample": True,
                    "fusion": "concat",
                    "multi_fuse": True,
                    "adapter": {
                        "type": "clearer",
                        "input_dim": 512,
                        "output_dim": 256,
                        "hidden_dim": 512,
                        "num_layers": 2,
                    },
                }
            }
        },
    }
    model = TSE_TFGRIDNET_VISUAL(config).eval()
    batch = {
        "wav_mix": torch.randn(2, 1, 16000),
        "visual_aux": torch.randn(2, 512, 25),
    }
    with torch.no_grad():
        output = model(batch)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameters / 1e6:.2f} M")
    print(f"Output: {tuple(output['speech'].shape)}")
