# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
"""BSRNN target extraction conditioned by audio and textual cues."""

import torch
import torch.nn as nn

from wesep.modules.common.deep_update import deep_update
from wesep.modules.separator.bsrnn import BSRNN
from wesep.modules.speaker.spk_frontend import SpeakerFrontend
from wesep.modules.textual.textual_frontend import TextualFrontend


class TSE_BSRNN_AUDIO_TEXTUAL(nn.Module):
    """Extract a target using speaker enrollment audio and KCE keywords."""

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
        if separator_config["channels"] != 1:
            raise ValueError(
                "TSE_BSRNN_AUDIO_TEXTUAL supports mono mixtures only")

        speaker_config = {
            "features": {
                "tfmap": {"enabled": True, "type": "spec"},
                "context": {
                    "enabled": False,
                    "mix_dim": separator_config["feature_dim"],
                    "atten_dim": separator_config["feature_dim"],
                },
                "spkemb": {
                    "enabled": False,
                    "input": "waveform",
                    "mix_dim": separator_config["feature_dim"],
                },
            },
        }
        speaker_config = deep_update(speaker_config, config.get("speaker", {}))
        separator_config["spec_dim"] = 2 + int(
            speaker_config["features"]["tfmap"]["enabled"])
        self.sep_model = BSRNN(**separator_config)
        self.spk_ft = SpeakerFrontend(speaker_config,
                                      defer_pretrained=defer_pretrained)

        textual_config = {
            "features": {
                "dae_kce": {
                    "enabled": True,
                    "mix_dim": separator_config["feature_dim"],
                    "num_stages": separator_config["num_repeat"],
                    "fusion": "FiLM",
                    "multi_fuse": True,
                }
            }
        }
        textual_config = deep_update(textual_config,
                                     config.get("textual", {}))
        textual_feature = textual_config["features"]["dae_kce"]
        textual_feature["mix_dim"] = separator_config["feature_dim"]
        textual_feature["num_stages"] = separator_config["num_repeat"]
        self.textual_ft = TextualFrontend(
            textual_config, defer_pretrained=defer_pretrained)

    def forward(self, batch):
        """Use wav_mix [B,1,T], audio_aux [B,1,T_e], textual_aux [B,L]."""
        mix = batch["wav_mix"]
        enroll = batch["audio_aux"]
        textual_aux = batch["textual_aux"]
        audio_present = batch.get("audio_aux_present")
        if mix.ndim == 2:
            mix = mix.unsqueeze(1)
        if mix.ndim != 3 or mix.shape[1] != 1:
            raise ValueError(f"Expected mono mixture [B,1,T], got {tuple(mix.shape)}")
        if self.spk_ft.input_type == "waveform" and enroll.ndim == 3:
            enroll = enroll.squeeze(1)

        mix_spec = self.sep_model.stft(mix)[-1]  # [B,1,F,T]
        reference_spec = mix_spec[:, 0]
        spectral_repr = torch.cat([mix_spec.real, mix_spec.imag], dim=1)

        if hasattr(self.spk_ft, "tfmap"):
            enroll_mag = self.sep_model.stft(enroll)[0]
            tfmap = self.spk_ft.tfmap.compute(
                enroll_mag,
                torch.abs(reference_spec),
                present=audio_present,
            )
            spectral_repr = self.spk_ft.tfmap.post(
                spectral_repr, tfmap.unsqueeze(1))

        subband_spec = self.sep_model.band_split(spectral_repr)
        subband_mix_spec = self.sep_model.band_split(reference_spec)
        subband_feature = self.sep_model.subband_norm(subband_spec)
        textual_cue = self.textual_ft.dae_kce.compute(mix[:, 0], textual_aux)

        sep_output = subband_feature
        if not self.textual_ft.dae_kce.multi_fuse:
            sep_output = self.textual_ft.dae_kce.post(
                sep_output, textual_cue, stage=0)
        for index in range(self.sep_model.separator.num_blocks):
            if self.textual_ft.dae_kce.multi_fuse:
                sep_output = self.textual_ft.dae_kce.post(
                    sep_output, textual_cue, stage=index)
            sep_output = self.sep_model.separator.forward_block(
                sep_output, index)

        estimated_ri = self.sep_model.band_masker(
            sep_output, subband_mix_spec)
        estimated_spec = torch.complex(estimated_ri[:, 0], estimated_ri[:, 1])
        speech = self.sep_model.istft(estimated_spec, length=mix.shape[-1])
        return {"speech": speech}
