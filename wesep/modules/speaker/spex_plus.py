# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Reference:
#   M. Ge et al., "SpEx+: A Complete Time Domain Speaker Extraction Network."

import torch
import torch.nn as nn

from wesep.modules.common.norm import select_norm


class ResidualSpeakerBlock(nn.Module):
    """Downsample enrollment features with a residual 1-D block."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, 1, bias=False)
        self.norm1 = nn.BatchNorm1d(output_dim)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(output_dim, output_dim, 1, bias=False)
        self.norm2 = nn.BatchNorm1d(output_dim)
        self.residual = (nn.Conv1d(input_dim, output_dim, 1, bias=False)
                         if input_dim != output_dim else nn.Identity())
        self.prelu2 = nn.PReLU()
        self.pool = nn.MaxPool1d(3)

    def forward(self, x):
        residual = self.residual(x)
        x = self.prelu1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.pool(self.prelu2(x + residual))


class SpExFusionBlock(nn.Module):
    """Fuse a speaker embedding into the first TCN block of one repeat."""

    def __init__(
        self,
        bottleneck_dim,
        embed_dim,
        conv_dim,
        kernel_size,
        causal,
        skip_connection,
    ):
        super().__init__()
        if not causal and kernel_size % 2 == 0:
            raise ValueError("Noncausal TCN blocks require an odd kernel size")

        norm_type = "cLN" if causal else "gLN"
        self.causal = causal
        self.padding = kernel_size - 1
        conv_padding = self.padding if causal else self.padding // 2

        self.input_proj = nn.Conv1d(bottleneck_dim + embed_dim, conv_dim, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm_type, conv_dim)
        self.depthwise_conv = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size,
            padding=conv_padding,
            groups=conv_dim,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm_type, conv_dim)
        self.residual_proj = nn.Conv1d(conv_dim, bottleneck_dim, 1)
        self.skip_proj = (nn.Conv1d(conv_dim, bottleneck_dim, 1)
                          if skip_connection else None)

    def forward(self, x, embedding):
        residual = x
        embedding = embedding.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.cat([x, embedding], dim=1)
        x = self.norm1(self.prelu1(self.input_proj(x)))
        x = self.depthwise_conv(x)
        if self.causal and self.padding:
            x = x[..., :-self.padding]
        x = self.norm2(self.prelu2(x))
        skip = self.skip_proj(x) if self.skip_proj is not None else None
        return residual + self.residual_proj(x), skip


class SpExPlusFeature(nn.Module):
    """Compute the SpEx+ speaker embedding and repeat-level fusion blocks."""

    def __init__(self, config):
        super().__init__()
        enc_dim = config["enc_dim"]
        embed_dim = config["embed_dim"]
        self.embedding_net = nn.Sequential(
            select_norm("cLN", 3 * enc_dim),
            nn.Conv1d(3 * enc_dim, enc_dim, 1),
            ResidualSpeakerBlock(enc_dim, enc_dim),
            ResidualSpeakerBlock(enc_dim, 2 * enc_dim),
            ResidualSpeakerBlock(2 * enc_dim, 2 * enc_dim),
            nn.Conv1d(2 * enc_dim, embed_dim, 1),
        )
        self.fusion_blocks = nn.ModuleList([
            SpExFusionBlock(
                bottleneck_dim=config["bottleneck_dim"],
                embed_dim=embed_dim,
                conv_dim=config["conv_dim"],
                kernel_size=config["kernel_size"],
                causal=config["causal"],
                skip_connection=config["skip_connection"],
            ) for _ in range(config["num_repeats"])
        ])

    @property
    def num_repeats(self):
        return len(self.fusion_blocks)

    def compute(self, enroll_repr, mix=None, present=None):
        """Pool encoded enrollment features `[B, 3N, K]` into `[B, D]`."""
        return self.embedding_net(enroll_repr).mean(dim=-1)

    def post(self, mix_repr, embedding, repeat_index, present=None):
        """Run the speaker-conditioned first block of one TCN repeat."""
        if not 0 <= repeat_index < self.num_repeats:
            raise IndexError(
                f"SpEx+ fusion repeat index out of range: {repeat_index}")
        return self.fusion_blocks[repeat_index](mix_repr, embedding)
