# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#               2026 Haoyu Li (haoyu.li.cs@sjtu.edu.cn)
#
# SPDX-License-Identifier: Apache-2.0
"""DPCCN target extraction conditioned by a textual keyword cue."""

import torch
import torch.nn as nn

from wesep.modules.common.deep_update import deep_update
from wesep.modules.separator.dpccn import DPCCN
from wesep.modules.textual.textual_frontend import TextualFrontend


class TSE_DPCCN_TEXTUAL(nn.Module):
    """Extract one target speaker with KCE-conditioned DPCCN.

    This preserves the speaker-conditioned DPCCN fusion location: the cue is
    fused immediately after the public pre-encoder stage. KCE replaces only
    the enrollment encoder and does not alter the separator architecture.
    """

    def __init__(self, config, defer_pretrained=False):
        super().__init__()

        separator_config = {
            "win": 512,
            "stride": 128,
            "nspk": 1,
            "tcn_layers": 2,
            "tcn_blocks": 10,
            "pool_sizes": (4, 8, 16, 32),
            "causal": False,
        }
        separator_config.update(config.get("separator", {}))
        if separator_config["nspk"] != 1:
            raise ValueError("TSE_DPCCN_TEXTUAL requires separator nspk=1")

        # The DPCCN speaker model fuses one utterance-level condition across
        # frequency bins after its input dense block. Keep that exact layout.
        textual_config = {
            "features": {
                "dae_kce": {
                    "enabled": True,
                    "mix_dim": separator_config["win"] // 2 + 1,
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
            raise ValueError("TSE_DPCCN_TEXTUAL requires dae_kce")
        if feature["multi_fuse"]:
            raise ValueError(
                "TSE_DPCCN_TEXTUAL supports one fusion after encode_input; "
                "set textual.features.dae_kce.multi_fuse=false")
        feature["mix_dim"] = separator_config["win"] // 2 + 1
        feature["num_stages"] = 1

        self.sep_model = DPCCN(**separator_config)
        self.textual_ft = TextualFrontend(
            textual_config,
            defer_pretrained=defer_pretrained,
        )

    def forward(self, batch):
        """Extract from mono ``wav_mix`` using ``textual_aux`` phoneme IDs."""
        mixture = batch["wav_mix"]
        textual_aux = batch["textual_aux"]
        if mixture.ndim == 3 and mixture.shape[1] == 1:
            mixture = mixture.squeeze(1)
        if mixture.ndim != 2:
            raise ValueError(
                "TSE_DPCCN_TEXTUAL expects mono mixture [B,1,T] or [B,T], "
                f"got {tuple(mixture.shape)}")
        length = mixture.shape[-1]

        # S1. Convert the mixture to stacked real and imaginary spectra.
        spectrum = self.sep_model.stft(mixture)[-1]  # [B, F, T]
        spec_ri = torch.stack(
            [spectrum.real, spectrum.imag],
            dim=1,
        ).transpose(2, 3)  # [B, 2, T, F]

        # S2. Preserve the DPCCN speaker fusion point after the input encoder.
        encoded_input = self.sep_model.encode_input(spec_ri)  # [B,16,T,F]
        textual_cue = self.textual_ft.dae_kce.compute(mixture, textual_aux)
        encoded_input = self.textual_ft.dae_kce.post(
            encoded_input.transpose(2, 3),  # [B,16,F,T]
            textual_cue,
            stage=0,
        ).transpose(2, 3).contiguous()  # [B,16,T,F]

        # S3. Run the unchanged DPCCN separator and decoder.
        bottleneck, skips = self.sep_model.encoder(encoded_input)
        separated = self.sep_model.separator(bottleneck)
        decoded = self.sep_model.decoder(separated, skips)
        decoded = self.sep_model.pyramid_projection(self.sep_model.pyramid(
            decoded))
        estimated_spectrum = self.sep_model.output_conv(decoded)
        speech = self.sep_model.reconstruct(estimated_spectrum, length)
        return {"speech": speech}
