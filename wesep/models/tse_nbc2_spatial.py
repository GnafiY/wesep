# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from wesep.modules.separator.nbc2 import NBC2
from wesep.modules.spatial.spatial_frontend import SpatialFrontend


class TSE_NBC2_SPATIAL(nn.Module):
    """Extract reference-channel speech with a narrow-band spatial model."""

    def __init__(self, config, defer_pretrained=False):
        super().__init__()

        separator_config = {
            "sr": 16000,
            "win": 512,
            "stride": 256,
            "channels": 1,
            "reference_channel": 0,
            "nspk": 1,
            "n_layers": 8,
            "dim_hidden": 96,
            "dim_ffn": 192,
            "block_kwargs": {
                "n_heads": 2,
                "dropout": 0.1,
                "conv_kernel_size": 3,
                "n_conv_groups": 8,
                "norms": ("LN", "GBN", "GBN"),
            },
        }
        separator_config.update(config.get("separator", {}))

        # Build the frontend first so its spectral width configures NBC2.
        self.spatial_ft = SpatialFrontend(
            config.get("spatial", {}),
            sample_rate=separator_config["sr"],
            n_fft=separator_config["win"],
            mix_dim=separator_config["dim_hidden"],
            num_mics=separator_config["channels"],
        )
        if hasattr(self.spatial_ft, "initstate_emb"):
            raise ValueError(
                "initstate_emb is only supported by recurrent separators")
        channels = separator_config["channels"]
        separator_config["input_size"] = (2 * channels +
                                          self.spatial_ft.spectral_channels)
        self.sep_model = NBC2(**separator_config)

    def forward(self, batch):
        """Use a multichannel mix and its target-direction cue."""
        mix = batch["wav_mix"]
        spatial_aux = batch["spatial_aux"]
        if mix.ndim != 3 or mix.shape[1] != self.sep_model.channels:
            raise ValueError(
                f"Expected [B, {self.sep_model.channels}, T] mixture, got "
                f"{tuple(mix.shape)}")

        # S1. Convert every input channel into the frequency domain.
        mix_spec = self.sep_model.stft(mix)[-1]  # [B, C, F, T]

        # S2. Normalize amplitude and concatenate real/imaginary components.
        mix_spec, scale = self.sep_model.normalize(mix_spec)
        spectral_repr = torch.cat(
            [mix_spec.real, mix_spec.imag],
            dim=1,
        )  # [B, 2*C, F, T]

        # C1. Append observed inter-channel phase differences.
        if hasattr(self.spatial_ft, "ipd"):
            ipd = self.spatial_ft.ipd.compute(mix_spec)  # [B, P, F, T]
            spectral_repr = self.spatial_ft.ipd.post(
                spectral_repr,
                ipd,
            )  # [B, D_s, F, T]

        # C2. Append direction-conditioned cosine phase features.
        if hasattr(self.spatial_ft, "cdf"):
            cdf = self.spatial_ft.cdf.compute(
                spatial_aux,
                mix_spec,
            )  # [B, P, F, T]
            spectral_repr = self.spatial_ft.cdf.post(
                spectral_repr,
                cdf,
            )  # [B, D_s, F, T]

        # C3. Append direction-conditioned sine phase features.
        if hasattr(self.spatial_ft, "sdf"):
            sdf = self.spatial_ft.sdf.compute(
                spatial_aux,
                mix_spec,
            )  # [B, P, F, T]
            spectral_repr = self.spatial_ft.sdf.post(
                spectral_repr,
                sdf,
            )  # [B, D_s, F, T]

        # C4. Append real and imaginary microphone-pair differences.
        if hasattr(self.spatial_ft, "delta_stft"):
            delta = self.spatial_ft.delta_stft.compute(mix_spec,
                                                       )  # [B, 2*P, F, T]
            spectral_repr = self.spatial_ft.delta_stft.post(
                spectral_repr,
                delta,
            )  # [B, D_s, F, T]

        # S3. Project each narrow-band input into the hidden dimension.
        hidden = self.sep_model.encoder(spectral_repr)  # [B, H, F, T]

        # C5. Compute the frame-level direction embedding once.
        doa_feature = None
        if hasattr(self.spatial_ft, "cyc_doaemb"):
            doa_feature = self.spatial_ft.cyc_doaemb.compute(
                spatial_aux, hidden)

        # S4. Model temporal structure independently at every frequency bin.
        for index in range(self.sep_model.num_layers):
            if doa_feature is not None:
                # Insert the direction embedding before every NBC2 block.
                hidden = hidden.permute(0, 2, 1, 3)
                hidden = self.spatial_ft.cyc_doaemb.post(hidden, doa_feature)
                hidden = hidden.permute(0, 2, 1, 3).contiguous()
            hidden = self.sep_model.separator.forward_block(
                hidden,
                index,
            )  # [B, H, F, T]

        # S5. Decode and restore the target complex spectra.
        estimated_spectrum = self.sep_model.decode_spectrum(
            hidden,
            scale,
        )  # [B, S, F, T]

        # S6. Convert every estimated source back into a waveform.
        speech = self.sep_model.istft(
            estimated_spectrum,
            length=mix.shape[-1],
        )  # [B, S, L]
        return {"speech": speech}


if __name__ == "__main__":
    config = {
        "separator": {
            "channels": 4,
            "n_layers": 2,
            "dim_hidden": 32,
            "dim_ffn": 64,
            "block_kwargs": {
                "n_heads": 2,
                "dropout": 0.0,
                "conv_kernel_size": 3,
                "n_conv_groups": 4,
                "norms": ("LN", "GBN", "GBN"),
            },
        },
        "spatial": {
            "input_fields": ["azimuth", "elevation"],
            "array": {
                "mic_positions": [
                    [0.02, 0.02, 0.0],
                    [-0.02, 0.02, 0.0],
                    [-0.02, -0.02, 0.0],
                    [0.02, -0.02, 0.0],
                ],
            },
            "pairs": [[0, 1], [1, 2], [2, 3], [0, 3]],
            "features": {
                "ipd": {
                    "enabled": True
                },
                "cdf": {
                    "enabled": True
                },
                "sdf": {
                    "enabled": True
                },
                "delta_stft": {
                    "enabled": True
                },
                "cyc_doaemb": {
                    "enabled": True,
                    "encoding_dim": 16,
                    "use_elevation": True,
                    "fusion": "multiply",
                },
                # Recurrent initial states are not available in NBC2.
                "initstate_emb": {
                    "enabled": False
                },
            },
        },
    }
    model = TSE_NBC2_SPATIAL(config).eval()
    batch = {
        "wav_mix": torch.randn(2, 4, 16000),
        "spatial_aux": torch.tensor([[0.4, 0.0], [1.2, 0.1]]),
    }
    with torch.no_grad():
        output = model(batch)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameters / 1e6:.2f} M")
    print(f"Output: {tuple(output['speech'].shape)}")
