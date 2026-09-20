# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from wesep.modules.separator.bsrnn import BSRNN
from wesep.modules.spatial.spatial_frontend import SpatialFrontend


class TSE_BSRNN_SPATIAL(nn.Module):
    """Extract reference-channel speech from a multichannel spatial mixture."""

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

        # Build the frontend first so its spectral width configures BSRNN.
        self.spatial_ft = SpatialFrontend(
            config.get("spatial", {}),
            sample_rate=separator_config["sr"],
            n_fft=separator_config["win"],
            mix_dim=separator_config["feature_dim"],
            num_layers=separator_config["num_repeat"],
            num_mics=separator_config["channels"],
        )
        channels = separator_config["channels"]
        separator_config["spec_dim"] = (2 * channels +
                                        self.spatial_ft.spectral_channels)
        self.sep_model = BSRNN(**separator_config)

    def forward(self, batch):
        """Use a multichannel mix and its target-direction cue."""
        mix = batch["wav_mix"]
        spatial_aux = batch["spatial_aux"]
        if mix.ndim != 3 or mix.shape[1] != self.sep_model.channels:
            raise ValueError(
                f"Expected [B, {self.sep_model.channels}, T] mixture, got "
                f"{tuple(mix.shape)}")

        # S1. Convert every microphone signal into a complex spectrum.
        mix_spec = self.sep_model.stft(mix)[-1]  # [B, C, F, T]
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

        # S2. Split the input and reference spectra into frequency bands.
        subband_spec = self.sep_model.band_split(spectral_repr,
                                                 )  # N_b * [B, D_s, BW, T]
        reference_spec = mix_spec[:, self.sep_model.reference_channel,
                                  ]  # [B, F, T]
        subband_mix_spec = self.sep_model.band_split(reference_spec,
                                                     )  # N_b * [B, BW, T]

        # S3. Normalize and project each band into the separator dimension.
        subband_feature = self.sep_model.subband_norm(subband_spec,
                                                      )  # [B, N_b, E, T]

        # C5. Fuse the frame-level cyclic direction embedding.
        if hasattr(self.spatial_ft, "cyc_doaemb"):
            doa_feature = self.spatial_ft.cyc_doaemb.compute(
                spatial_aux, subband_feature)
            subband_feature = self.spatial_ft.cyc_doaemb.post(
                subband_feature, doa_feature)

        # C6. Generate optional direction-conditioned recurrent states.
        initial_states = None
        if hasattr(self.spatial_ft, "initstate_emb"):
            initial_states = self.spatial_ft.initstate_emb.compute(spatial_aux)

        # S4. Separate the target representation across bands and frames.
        sep_output = self.sep_model.separator(
            subband_feature,
            initial_states=initial_states,
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
    """Report when future mixture changes first affect causal output."""
    sample_rate = model.sep_model.sr
    mix = torch.randn(1, model.sep_model.channels, sample_rate * 4)
    spatial_aux = torch.tensor([[0.4, 0.0]])
    batch = {"wav_mix": mix, "spatial_aux": spatial_aux}

    model = model.eval()
    with torch.no_grad():
        reference = model(batch)["speech"]
        for second in range(1, 3):
            changed_mix = mix.clone()
            changed_mix[..., second * sample_rate:] = torch.randn_like(
                changed_mix[..., second * sample_rate:])
            changed = model({**batch, "wav_mix": changed_mix})["speech"]
            first_change = ((reference - changed).abs() >
                            1e-8).flatten().float().argmax()
            print(
                f"input={second:.1f}s, output={first_change / sample_rate:.3f}s"
            )


if __name__ == "__main__":
    config = {
        "separator": {
            "feature_dim": 32,
            "num_repeat": 2,
            "causal": True,
            "channels": 4,
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
                "initstate_emb": {
                    "enabled": True,
                    "encoding_dim": 16,
                    "use_elevation": True,
                },
            },
        },
    }
    model = TSE_BSRNN_SPATIAL(config).eval()
    batch = {
        "wav_mix": torch.randn(2, 4, 16000),
        "spatial_aux": torch.tensor([[0.4, 0.0], [1.2, 0.1]]),
    }
    with torch.no_grad():
        output = model(batch)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameters / 1e6:.2f} M")
    print(f"Output: {tuple(output['speech'].shape)}")
    check_causal(model)
