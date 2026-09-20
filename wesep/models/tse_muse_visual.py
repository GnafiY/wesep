# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
"""MuSE target-speaker extraction adapted to the WeSep model interface.

The visual input is expected to be the frame-level 512-dimensional feature
produced by the frozen MuSE visual frontend.  The trainable visual TCN and the
iterative speaker-conditioned extraction network remain part of this model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.visual.muse import Muse_VisualConv1D


class ChannelWiseLayerNorm(nn.LayerNorm):
    """Apply LayerNorm to the channel dimension of a [B, C, T] tensor."""

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected a 3-D tensor, got {tuple(x.shape)}")
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class GlobalLayerNorm(nn.Module):
    """Global layer normalization used by the original MuSE TCN."""

    def __init__(self, channels, eps=1.0e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        variance = (x - mean).square().mean(dim=(1, 2), keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(variance +
                                                     self.eps) + self.bias


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, channels, output_channels, kernel_size, padding,
                 dilation, norm_eps):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.PReLU(),
            GlobalLayerNorm(channels, eps=norm_eps),
            nn.Conv1d(channels, output_channels, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class TemporalBlock(nn.Module):

    def __init__(self, channels, hidden_dim, kernel_size, dilation, norm_eps):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, 1, bias=False),
            nn.PReLU(),
            GlobalLayerNorm(hidden_dim, eps=norm_eps),
            DepthwiseSeparableConv(
                hidden_dim,
                channels,
                kernel_size,
                padding,
                dilation,
                norm_eps,
            ),
        )

    def forward(self, x):
        return x + self.net(x)


class SpeakerEmbedding(nn.Module):
    """Estimate one self-enrolled speaker representation per MuSE repeat."""

    def __init__(self, bottleneck_dim, hidden_dim=256, num_blocks=3):
        super().__init__()
        self.input_projection = nn.Conv1d(2 * bottleneck_dim,
                                          bottleneck_dim,
                                          1,
                                          bias=False)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(bottleneck_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.PReLU(),
                nn.Conv1d(hidden_dim, bottleneck_dim, 1, bias=False),
                nn.BatchNorm1d(bottleneck_dim),
            ) for _ in range(num_blocks)
        ])
        self.activations = nn.ModuleList(
            [nn.PReLU() for _ in range(num_blocks)])
        self.pools = nn.ModuleList(
            [nn.AvgPool1d(3) for _ in range(num_blocks)])
        self.output_projection = nn.Conv1d(bottleneck_dim, bottleneck_dim, 1)
        self.average = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.input_projection(x)
        for block, activation, pool in zip(self.blocks, self.activations,
                                           self.pools):
            x = pool(activation(block(x) + x))
        return self.average(self.output_projection(x))


class MuSESeparator(nn.Module):

    def __init__(
        self,
        encoder_dim,
        bottleneck_dim,
        hidden_dim,
        kernel_size,
        blocks_per_repeat,
        num_repeats,
        visual_dim,
        visual_layers,
        visual_upsample_ratio,
        speaker_embedding_hidden_dim,
        speaker_embedding_blocks,
        norm_eps,
        num_speakers=None,
    ):
        super().__init__()
        self.num_repeats = num_repeats
        self.visual_upsample_ratio = visual_upsample_ratio

        self.input_norm = ChannelWiseLayerNorm(encoder_dim, eps=norm_eps)
        self.bottleneck = nn.Conv1d(encoder_dim, bottleneck_dim, 1, bias=False)

        self.visual_tcn = nn.Sequential(*[
            Muse_VisualConv1D(channels=visual_dim)
            for _ in range(visual_layers)
        ])
        self.visual_projections = nn.ModuleList([
            nn.Conv1d(visual_dim, bottleneck_dim, 1, bias=False)
            for _ in range(num_repeats)
        ])
        self.speaker_visual_projections = nn.ModuleList([
            nn.Conv1d(visual_dim, bottleneck_dim, 1, bias=False)
            for _ in range(num_repeats)
        ])
        self.speaker_encoders = nn.ModuleList([
            SpeakerEmbedding(
                bottleneck_dim,
                hidden_dim=speaker_embedding_hidden_dim,
                num_blocks=speaker_embedding_blocks,
            ) for _ in range(num_repeats)
        ])

        repeats = []
        for _ in range(num_repeats):
            layers = [
                nn.Conv1d(3 * bottleneck_dim, bottleneck_dim, 1, bias=False)
            ]
            layers.extend([
                TemporalBlock(
                    bottleneck_dim,
                    hidden_dim,
                    kernel_size,
                    dilation=2**block_index,
                    norm_eps=norm_eps,
                ) for block_index in range(blocks_per_repeat)
            ])
            repeats.append(nn.Sequential(*layers))
        self.repeats = nn.ModuleList(repeats)

        self.speaker_classifiers = None
        if num_speakers is not None:
            self.speaker_classifiers = nn.ModuleList([
                nn.Linear(bottleneck_dim, num_speakers)
                for _ in range(num_repeats)
            ])

        self.mask_projection = nn.Conv1d(bottleneck_dim,
                                         encoder_dim,
                                         1,
                                         bias=False)

    def _align_visual(self, visual, audio_frames):
        target_frames = self.visual_upsample_ratio * visual.shape[-1]
        visual = F.interpolate(
            visual,
            size=target_frames,
            mode="linear",
            align_corners=False,
        )
        if target_frames < audio_frames:
            visual = F.pad(visual, (0, audio_frames - target_frames))
        elif target_frames > audio_frames:
            visual = visual[..., :audio_frames]
        return visual

    def forward(self, encoded_mixture, visual):
        visual = self.visual_tcn(visual)
        x = self.bottleneck(self.input_norm(encoded_mixture))
        mixture = x
        audio_frames = x.shape[-1]
        speaker_logits = []

        for index in range(self.num_repeats):
            separation_visual = self._align_visual(
                self.visual_projections[index](visual), audio_frames)
            speaker_visual = self._align_visual(
                self.speaker_visual_projections[index](visual), audio_frames)

            rough_target = mixture * F.relu(x)
            speaker_embedding = self.speaker_encoders[index](torch.cat(
                [rough_target, speaker_visual], dim=1))

            if self.speaker_classifiers is not None:
                speaker_logits.append(self.speaker_classifiers[index](
                    speaker_embedding.squeeze(-1)))

            repeated_embedding = speaker_embedding.expand(-1, -1, audio_frames)
            x = self.repeats[index](torch.cat([
                repeated_embedding,
                x,
                separation_visual,
            ],
                                              dim=1))

        mask = F.relu(self.mask_projection(x))
        return mask, speaker_logits


class MuSEDecoder(nn.Module):

    def __init__(self, encoder_dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size // 2
        self.basis = nn.Linear(encoder_dim, kernel_size, bias=False)

    def forward(self, encoded_mixture, mask):
        frames = self.basis((encoded_mixture * mask).transpose(1, 2))
        num_frames = frames.shape[1]
        output_length = self.stride * (num_frames - 1) + self.kernel_size
        waveform = F.fold(
            frames.transpose(1, 2),
            output_size=(1, output_length),
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
        )
        return waveform.squeeze(2)


class TSE_MUSE_VISUAL(nn.Module):
    """Original MuSE extraction path with WeSep batch-dict I/O."""

    def __init__(self, config, defer_pretrained=False):
        super().__init__()
        del defer_pretrained  # The frozen frame frontend has run offline.

        encoder_dim = config.get("encoder_dim", 256)
        kernel_size = config.get("kernel_size", 40)
        bottleneck_dim = config.get("bottleneck_dim", 256)
        hidden_dim = config.get("hidden_dim", 512)
        tcn_kernel_size = config.get("tcn_kernel_size", 3)
        blocks_per_repeat = config.get("blocks_per_repeat", 8)
        num_repeats = config.get("num_repeats", 4)
        visual_dim = config.get("visual_dim", 512)
        visual_layers = config.get("visual_layers", 5)
        visual_upsample_ratio = config.get("visual_upsample_ratio", 32)
        norm_eps = config.get("norm_eps", 1.0e-8)

        speaker_embedding = config.get("speaker_embedding", {})
        identity = config.get("speaker_identity", {})
        identity_enabled = identity.get("enabled", False)
        num_speakers = identity.get("num_speakers")
        if identity_enabled and (num_speakers is None or num_speakers <= 0):
            raise ValueError(
                "speaker_identity.num_speakers must be positive when enabled")
        if not identity_enabled:
            num_speakers = None

        self.visual_dim = visual_dim
        self.encoder = nn.Conv1d(
            1,
            encoder_dim,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False,
        )
        self.separator = MuSESeparator(
            encoder_dim=encoder_dim,
            bottleneck_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
            kernel_size=tcn_kernel_size,
            blocks_per_repeat=blocks_per_repeat,
            num_repeats=num_repeats,
            visual_dim=visual_dim,
            visual_layers=visual_layers,
            visual_upsample_ratio=visual_upsample_ratio,
            speaker_embedding_hidden_dim=speaker_embedding.get(
                "hidden_dim", 256),
            speaker_embedding_blocks=speaker_embedding.get("num_blocks", 3),
            norm_eps=norm_eps,
            num_speakers=num_speakers,
        )
        self.decoder = MuSEDecoder(encoder_dim, kernel_size)

        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_normal_(parameter)

    def forward(self, batch):
        mixture = batch["wav_mix"]
        visual = batch["visual_aux"]
        if mixture.ndim == 2:
            mixture = mixture.unsqueeze(1)
        if mixture.ndim != 3 or mixture.shape[1] != 1:
            raise ValueError(
                f"MuSE expects mono mixture [B, 1, T], got {tuple(mixture.shape)}"
            )
        if visual.ndim != 3 or visual.shape[1] != self.visual_dim:
            raise ValueError(
                "MuSE expects precomputed visual features "
                f"[B, {self.visual_dim}, Tv], got {tuple(visual.shape)}")
        if mixture.shape[0] != visual.shape[0]:
            raise ValueError("Mixture and visual batch sizes do not match")

        input_length = mixture.shape[-1]
        encoded_mixture = F.relu(self.encoder(mixture))
        mask, speaker_logits = self.separator(encoded_mixture, visual)
        speech = self.decoder(encoded_mixture, mask)
        if speech.shape[-1] < input_length:
            speech = F.pad(speech, (0, input_length - speech.shape[-1]))
        elif speech.shape[-1] > input_length:
            speech = speech[..., :input_length]

        outputs = {"speech": speech}
        for index, logits in enumerate(speaker_logits):
            outputs[f"speaker_logits_{index}"] = logits
        return outputs


if __name__ == "__main__":
    model = TSE_MUSE_VISUAL({
        "encoder_dim": 32,
        "bottleneck_dim": 16,
        "hidden_dim": 32,
        "blocks_per_repeat": 2,
        "num_repeats": 2,
        "visual_dim": 32,
        "visual_layers": 2,
    })
    example = {
        "wav_mix": torch.randn(2, 1, 16000),
        "visual_aux": torch.randn(2, 32, 25),
    }
    result = model(example)
    result["speech"].square().mean().backward()
    print(tuple(result["speech"].shape))
