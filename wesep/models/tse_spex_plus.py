# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Reference:
#   M. Ge et al., "SpEx+: A Complete Time Domain Speaker Extraction Network."

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.common.deep_update import deep_update
from wesep.modules.common.norm import select_norm
from wesep.modules.separator.convtasnet import TCNSeparator
from wesep.modules.speaker.spk_frontend import SpeakerFrontend


class MultiScaleEncoder(nn.Module):
    """Encode one waveform into aligned short, middle, and long features."""

    def __init__(
        self,
        enc_dim,
        bottleneck_dim,
        kernel_sizes,
        stride,
    ):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.short_kernel = self.kernel_sizes[0]
        self.encoders = nn.ModuleList([
            nn.Conv1d(1, enc_dim, kernel_size, stride=stride, bias=True)
            for kernel_size in self.kernel_sizes
        ])
        self.norm = select_norm("cLN", len(self.kernel_sizes) * enc_dim)
        self.proj = nn.Conv1d(
            len(self.kernel_sizes) * enc_dim, bottleneck_dim, 1)

    def forward(self, x):
        scale_features = []
        for kernel_size, encoder in zip(self.kernel_sizes, self.encoders):
            scale_input = F.pad(x, (0, kernel_size - self.short_kernel))
            scale_features.append(F.relu(encoder(scale_input)))  # [B, N, K]
        bottleneck = self.proj(self.norm(torch.cat(scale_features, dim=1)),
                               )  # [B, H, K]
        return bottleneck, scale_features


class MultiScaleDecoder(nn.Module):
    """Estimate and decode one target waveform at each encoder scale."""

    def __init__(
        self,
        bottleneck_dim,
        enc_dim,
        kernel_sizes,
        stride,
        activation="relu",
    ):
        super().__init__()
        if activation not in {"relu", "sigmoid"}:
            raise ValueError(
                f"Unsupported SpEx+ mask activation: {activation}")
        self.activation = activation
        self.maskers = nn.ModuleList(
            [nn.Conv1d(bottleneck_dim, enc_dim, 1) for _ in kernel_sizes])
        self.decoders = nn.ModuleList([
            nn.ConvTranspose1d(enc_dim,
                               1,
                               kernel_size,
                               stride=stride,
                               bias=True) for kernel_size in kernel_sizes
        ])

    def forward(self, x, scale_features, length):
        estimates = []
        for masker, decoder, encoded in zip(self.maskers, self.decoders,
                                            scale_features):
            mask = masker(x)  # [B, N, K]
            mask = F.relu(
                mask) if self.activation == "relu" else mask.sigmoid()
            estimate = decoder(mask * encoded)[..., :length]  # [B, 1, L]
            estimates.append(estimate)
        return estimates


class TSE_SPEX_PLUS(nn.Module):
    """Extract one target speaker with the multi-scale SpEx+ architecture."""

    def __init__(self, config, defer_pretrained=False):
        super().__init__()

        separator_config = {
            "enc_dim": 256,
            "kernel_sizes": [20, 80, 160],
            "stride": 10,
            "bottleneck_dim": 256,
            "conv_dim": 512,
            "conv_kernel_size": 3,
            "num_blocks": 8,
            "num_repeats": 4,
            "causal": False,
            "skip_connection": False,
            "mask_activation": "relu",
        }
        separator_config.update(config.get("separator", {}))
        if len(separator_config["kernel_sizes"]) != 3:
            raise ValueError("SpEx+ requires exactly three encoder scales")
        if separator_config["num_blocks"] < 2:
            raise ValueError(
                "SpEx+ requires at least two TCN blocks per repeat")

        speaker_config = {
            "features": {
                "spex_plus": {
                    "enabled": True,
                    "input": "waveform",
                    "enc_dim": separator_config["enc_dim"],
                    "bottleneck_dim": separator_config["bottleneck_dim"],
                    "conv_dim": separator_config["conv_dim"],
                    "kernel_size": separator_config["conv_kernel_size"],
                    "num_repeats": separator_config["num_repeats"],
                    "embed_dim": 256,
                    "causal": separator_config["causal"],
                    "skip_connection": separator_config["skip_connection"],
                },
            },
        }
        self.speaker_config = deep_update(speaker_config,
                                          config.get("speaker", {}))
        if not self.speaker_config["features"]["spex_plus"]["enabled"]:
            raise ValueError("TSE_SPEX_PLUS requires the spex_plus feature")

        # Share one encoder between mixture and enrollment waveforms.
        self.encoder = MultiScaleEncoder(
            enc_dim=separator_config["enc_dim"],
            bottleneck_dim=separator_config["bottleneck_dim"],
            kernel_sizes=separator_config["kernel_sizes"],
            stride=separator_config["stride"],
        )
        self.spk_ft = SpeakerFrontend(
            self.speaker_config,
            defer_pretrained=defer_pretrained,
        )
        self.separator = TCNSeparator(
            num_repeats=separator_config["num_repeats"],
            num_blocks=separator_config["num_blocks"] - 1,
            bottleneck_dim=separator_config["bottleneck_dim"],
            conv_dim=separator_config["conv_dim"],
            kernel_size=separator_config["conv_kernel_size"],
            dilation_start=1,
            causal=separator_config["causal"],
            skip_connection=separator_config["skip_connection"],
        )
        self.decoder = MultiScaleDecoder(
            bottleneck_dim=separator_config["bottleneck_dim"],
            enc_dim=separator_config["enc_dim"],
            kernel_sizes=separator_config["kernel_sizes"],
            stride=separator_config["stride"],
            activation=separator_config["mask_activation"],
        )

        num_speakers = config.get("num_speakers")
        self.speaker_classifier = None
        if num_speakers is not None:
            embed_dim = self.speaker_config["features"]["spex_plus"][
                "embed_dim"]
            self.speaker_classifier = nn.Linear(embed_dim, num_speakers)

        self.short_kernel = separator_config["kernel_sizes"][0]
        self.stride = separator_config["stride"]

    def pad_input(self, x):
        """Right-pad `[B, 1, T]` for the shortest encoder scale."""
        length = x.shape[-1]
        padding = max(self.short_kernel - length, 0)
        padded_length = length + padding
        padding += (
            self.stride -
            (padded_length - self.short_kernel) % self.stride) % self.stride
        return F.pad(x, (0, padding)), length

    def forward(self, batch):
        mixture = batch["wav_mix"]
        enrollment = batch["audio_aux"]
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)
        if enrollment.dim() == 2:
            enrollment = enrollment.unsqueeze(1)
        if mixture.dim() != 3 or mixture.shape[1] != 1:
            raise ValueError(f"SpEx+ expects mono mixture [B, 1, T], got "
                             f"{tuple(mixture.shape)}")
        if enrollment.dim() != 3 or enrollment.shape[1] != 1:
            raise ValueError(f"SpEx+ expects mono enrollment [B, 1, T], got "
                             f"{tuple(enrollment.shape)}")

        mixture, mixture_length = self.pad_input(mixture)
        enrollment, _ = self.pad_input(enrollment)

        # S1. Encode the mixture at three temporal resolutions.
        separated, mixture_scales = self.encoder(mixture)  # [B,H,K], 3*[B,N,K]
        # C1. Encode the enrollment with the shared multi-scale encoder.
        _, enrollment_scales = self.encoder(enrollment)
        enrollment_repr = torch.cat(enrollment_scales, dim=1)  # [B, 3*N, K_e]
        speaker_embedding = self.spk_ft.spex_plus.compute(enrollment_repr,
                                                          )  # [B, D]

        # S2/C2. Fuse the speaker at each repeat before ordinary TCN blocks.
        skip_sum = None
        for repeat_index in range(self.separator.num_repeats):
            separated, fusion_skip = self.spk_ft.spex_plus.post(
                separated,
                speaker_embedding,
                repeat_index,
            )  # [B, H, K]
            separated, repeat_skip = self.separator.forward_repeat(
                separated,
                repeat_index,
            )  # [B, H, K]
            for skip in (fusion_skip, repeat_skip):
                if skip is not None:
                    skip_sum = skip if skip_sum is None else skip_sum + skip
        separated = skip_sum if skip_sum is not None else separated

        # S3. Decode short, middle, and long estimates of the same target.
        short, middle, long = self.decoder(
            separated,
            mixture_scales,
            mixture_length,
        )  # 3 * [B, 1, L]
        outputs = {
            "speech": short,
            "speech_middle": middle,
            "speech_long": long,
        }
        if self.speaker_classifier is not None:
            outputs["speaker_logits"] = self.speaker_classifier(
                speaker_embedding)
        return outputs


def check_causal(model, num_samples=16000, change_sample=8000):
    """Report the first output sample affected by future mixture changes."""
    mixture = torch.randn(1, 1, num_samples)
    changed = mixture.clone()
    changed[..., change_sample:] = torch.randn_like(changed[...,
                                                            change_sample:])
    enrollment = torch.randn(1, 1, num_samples // 2)
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
            "enc_dim": 32,
            "kernel_sizes": [16, 32, 64],
            "stride": 8,
            "bottleneck_dim": 16,
            "conv_dim": 32,
            "num_blocks": 4,
            "num_repeats": 2,
        },
        "speaker": {
            "features": {
                "spex_plus": {
                    "enabled": True,
                    "embed_dim": 16
                },
            },
        },
    }
    model = TSE_SPEX_PLUS(test_config).eval()
    test_batch = {
        "wav_mix": torch.randn(2, 1, 16003),
        "audio_aux": torch.randn(2, 1, 12001),
    }
    with torch.no_grad():
        test_outputs = model(test_batch)
    print({key: tuple(value.shape) for key, value in test_outputs.items()})
    print(f"First future-affected sample: {check_causal(model)}")
