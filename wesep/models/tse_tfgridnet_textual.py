# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#               2026 Haoyu Li (haoyu.li.cs@sjtu.edu.cn)
#
# SPDX-License-Identifier: Apache-2.0
"""TFGridNet target extraction conditioned by a textual keyword cue."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.common.deep_update import deep_update
from wesep.modules.separator.tfgridnet import TFGridNet
from wesep.modules.textual.textual_frontend import TextualFrontend


class TSE_TFGRIDNET_TEXTUAL(nn.Module):
    """Extract one target speaker with KCE-conditioned TFGridNet.

    KCE replaces the original utterance-level speaker embedding exactly
    at its post-input-projection fusion point; the separator is unchanged.
    """

    def __init__(self, config, defer_pretrained=False):
        super().__init__()

        separator_config = {
            "nspk": 1,
            "n_fft": 128,
            "stride": 64,
            "channels": 1,
            "n_layers": 6,
            "hidden_channels": 192,
            "n_head": 4,
            "approx_qk_dim": 512,
            "emb_dim": 48,
            "emb_ks": 4,
            "emb_hs": 1,
            "eps": 1e-5,
            "causal": False,
            "estimation": "spec",
            "reference_channel": 0,
            "attention_context": None,
        }
        separator_config.update(config.get("separator", {}))
        if separator_config["nspk"] != 1:
            raise ValueError(
                "TSE_TFGRIDNET_TEXTUAL requires separator nspk=1")
        separator_config["input_dim"] = 2 * separator_config["channels"]

        textual_config = {
            "features": {
                "dae_kce": {
                    "enabled": True,
                    "mix_dim": separator_config["emb_dim"],
                    "num_stages": 1,
                    "fusion": "multiply",
                    "multi_fuse": False,
                },
            },
        }
        textual_config = deep_update(textual_config,
                                     config.get("textual", {}))
        feature = textual_config["features"]["dae_kce"]
        if not feature["enabled"]:
            raise ValueError("TSE_TFGRIDNET_TEXTUAL requires dae_kce")
        if feature["multi_fuse"]:
            raise ValueError(
                "TSE_TFGRIDNET_TEXTUAL preserves one post-projection fusion; "
                "set textual.features.dae_kce.multi_fuse=false")
        feature["mix_dim"] = separator_config["emb_dim"]
        feature["num_stages"] = 1

        self.sep_model = TFGridNet(**separator_config)
        self.textual_ft = TextualFrontend(
            textual_config,
            defer_pretrained=defer_pretrained,
        )

    def forward(self, batch):
        """Extract from ``wav_mix`` using ``textual_aux`` phoneme IDs."""
        mixture = batch["wav_mix"]
        textual_aux = batch["textual_aux"]
        if mixture.ndim == 2 and self.sep_model.channels == 1:
            mixture = mixture.unsqueeze(1)
        if mixture.ndim != 3 or mixture.shape[1] != self.sep_model.channels:
            raise ValueError(
                f"Expected [B,{self.sep_model.channels},T] mixture, got "
                f"{tuple(mixture.shape)}")

        # KCE is defined for one waveform. For a multichannel TFGridNet, use
        # the same reference channel used to reconstruct masked estimates.
        reference_waveform = mixture[:, self.sep_model.reference_channel]
        mix_len = mixture.shape[-1]
        if self.sep_model.causal:
            scale = mixture.new_ones(mixture.shape[0], 1, 1)
        else:
            scale = mixture.std(dim=(1, 2), keepdim=True).clamp_min(
                self.sep_model.eps)
        spectrum = self.sep_model.stft(mixture / scale)[-1]  # [B,C,F,T]
        reference_spectrum = spectrum[:, self.sep_model.reference_channel]
        spec_ri = torch.cat([spectrum.real, spectrum.imag], dim=1)
        spec_ri = spec_ri.permute(0, 1, 3, 2).contiguous()  # [B,2C,T,F]
        if self.sep_model.causal:
            spec_ri = F.pad(spec_ri, (0, 0, 2, 0))
        features = self.sep_model.input_norm(
            self.sep_model.input_conv(spec_ri))  # [B,E,T,F]

        # Preserve the original speaker-embedding fusion representation.
        textual_cue = self.textual_ft.dae_kce.compute(reference_waveform,
                                                       textual_aux)
        fusion_features = features.permute(0, 3, 1, 2).contiguous()
        fusion_features = self.textual_ft.dae_kce.post(
            fusion_features,
            textual_cue,
            stage=0,
        )
        features = fusion_features.permute(0, 2, 3, 1).contiguous()
        features = self.sep_model.separator(features)

        estimated_ri = self.sep_model.output_conv(features)
        if self.sep_model.causal:
            estimated_ri = estimated_ri[..., :features.shape[-2], :]
        batch_size, _, frames, frequencies = estimated_ri.shape
        estimated_ri = estimated_ri.view(
            batch_size,
            self.sep_model.nspk,
            2,
            frames,
            frequencies,
        )
        # torch.complex cannot combine BF16 components from autocast.
        estimated_ri = estimated_ri.float()
        if self.sep_model.estimation == "mask":
            estimated_ri = 5.0 * torch.tanh(estimated_ri / 5.0)
            complex_mask = torch.complex(estimated_ri[:, :, 0],
                                         estimated_ri[:, :, 1])
            estimated_spectrum = (
                complex_mask *
                reference_spectrum.transpose(-2, -1).unsqueeze(1))
        else:
            estimated_spectrum = torch.complex(estimated_ri[:, :, 0],
                                               estimated_ri[:, :, 1])
        estimated_spectrum = estimated_spectrum.transpose(-2, -1).contiguous()
        speech = self.sep_model.istft(
            estimated_spectrum,
            length=mix_len,
        ) * scale
        return {"speech": speech}
