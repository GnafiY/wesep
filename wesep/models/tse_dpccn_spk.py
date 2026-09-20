# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Reference:
#   J. Han et al., "DPCCN: Densely-Connected Pyramid Complex Convolutional
#   Network for Robust Speech Separation and Extraction," ICASSP 2022.

import torch
import torch.nn as nn

from wesep.modules.common.deep_update import deep_update
from wesep.modules.separator.dpccn import DPCCN
from wesep.modules.speaker.spk_frontend import SpeakerFrontend


class TSE_DPCCN_SPK(nn.Module):
    """Extract one target speaker with speaker-conditioned DPCCN."""

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
            raise ValueError("TSE_DPCCN_SPK requires separator nspk=1")

        speaker_config = {
            "features": {
                "spkemb": {
                    "enabled": True,
                    "input": "waveform",
                    "fusion": "multiply",
                    "mix_dim": separator_config["win"] // 2 + 1,
                },
            },
        }
        self.speaker_config = deep_update(speaker_config,
                                          config.get("speaker", {}))
        if not self.speaker_config["features"]["spkemb"]["enabled"]:
            raise ValueError("TSE_DPCCN_SPK requires the spkemb feature")
        unsupported = [
            name for name, feature in self.speaker_config["features"].items()
            if name != "spkemb" and feature["enabled"]
        ]
        if unsupported:
            raise ValueError(
                f"TSE_DPCCN_SPK does not implement speaker features: "
                f"{unsupported}")
        self.speaker_config["features"]["spkemb"]["mix_dim"] = (
            separator_config["win"] // 2 + 1)

        self.sep_model = DPCCN(**separator_config)
        self.spk_ft = SpeakerFrontend(
            self.speaker_config,
            defer_pretrained=defer_pretrained,
        )

    def forward(self, batch):
        """Extract speech from mono `wav_mix` using `audio_aux`."""
        mixture = batch["wav_mix"]
        enrollment = batch["audio_aux"]
        if mixture.ndim == 3 and mixture.shape[1] == 1:
            mixture = mixture.squeeze(1)
        if mixture.ndim != 2:
            raise ValueError(
                f"TSE_DPCCN_SPK expects mono mixture [B,1,T] or [B,T], "
                f"got {tuple(mixture.shape)}")
        if self.spk_ft.input_type == "waveform":
            if enrollment.ndim == 3 and enrollment.shape[1] == 1:
                enrollment = enrollment.squeeze(1)
            if enrollment.ndim != 2:
                raise ValueError(
                    "Waveform enrollment must have shape [B,1,T] or [B,T], "
                    f"got {tuple(enrollment.shape)}")
        length = mixture.shape[-1]

        # S1. Convert the mixture into stacked real and imaginary spectra.
        spectrum = self.sep_model.stft(mixture)[-1]  # [B, F, T]
        spec_ri = torch.stack(
            [spectrum.real, spectrum.imag],
            dim=1,
        )  # [B, 2, F, T]
        spec_ri = spec_ri.transpose(2, 3)  # [B, 2, T, F]
        # S2. Run the first dense encoder stage exposed by DPCCN.
        encoded_input = self.sep_model.encode_input(spec_ri)  # [B,16,T,F]
        # C1. Extract one utterance-level speaker embedding.
        speaker_embedding = self.spk_ft.spkemb.compute(enrollment)  # [B,D]
        if speaker_embedding is not None:
            speaker_embedding = speaker_embedding.unsqueeze(1).unsqueeze(3)
        # C2. Fuse the embedding across the encoded frequency representation.
        encoded_input = self.spk_ft.spkemb.post(
            encoded_input.transpose(2, 3),
            speaker_embedding,
        ).transpose(2, 3)  # [B, 16, T, F]
        # S3. Finish dense encoding and model the time-frequency bottleneck.
        bottleneck, skips = self.sep_model.encoder(encoded_input,
                                                   )  # [B, 384, T, F_b]
        separated = self.sep_model.separator(bottleneck)  # [B,384,T,F_b]
        # S4. Decode with mirrored skip connections.
        decoded = self.sep_model.decoder(separated, skips)  # [B,32,T,F]
        # S5. Aggregate pyramid context and estimate the target complex spectrum.
        decoded = self.sep_model.pyramid_projection(
            self.sep_model.pyramid(decoded), )  # [B, 32, T, F]
        estimated_spectrum = self.sep_model.output_conv(decoded,
                                                        )  # [B, 2, T, F]
        # S6. Reconstruct the target waveform at the original mixture length.
        speech = self.sep_model.reconstruct(estimated_spectrum,
                                            length)  # [B,1,L]
        return {"speech": speech}


def check_causal(model, num_samples=16000, change_sample=8000):
    """Report the first output sample affected by future mixture changes."""
    mixture = torch.randn(1, 1, num_samples)
    changed = mixture.clone()
    changed[..., change_sample:] = torch.randn_like(changed[...,
                                                            change_sample:])
    enrollment = torch.randn(1, 192)
    model = model.eval()
    with torch.no_grad():
        output = model({
            "wav_mix": mixture,
            "audio_aux": enrollment,
        })["speech"]
        changed_output = model({
            "wav_mix": changed,
            "audio_aux": enrollment,
        })["speech"]
    difference = (output - changed_output).abs().amax(dim=1).squeeze(0)
    affected = torch.nonzero(difference > 1e-6)
    return affected[0].item() if affected.numel() else None


if __name__ == "__main__":
    test_config = {
        "separator": {
            "tcn_layers": 1,
            "tcn_blocks": 2,
            "causal": True,
        },
        "speaker": {
            "features": {
                "spkemb": {
                    "enabled": True,
                    "input": "embedding",
                },
            },
        },
    }
    model = TSE_DPCCN_SPK(test_config).eval()
    test_batch = {
        "wav_mix": torch.randn(2, 1, 16003),
        "audio_aux": torch.randn(2, 192),
    }
    with torch.no_grad():
        test_output = model(test_batch)
    print({key: tuple(value.shape) for key, value in test_output.items()})
    print(f"First future-affected sample: {check_causal(model)}")
