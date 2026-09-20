# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Reference:
#   P. Shen et al., "Listen to Extract: Onset-Prompted Target Speaker
#   Extraction," arXiv:2505.05114, 2025.

import torch
import torch.nn as nn


class ListenFeature(nn.Module):
    """Prepend an enrollment utterance and a glue signal to the mixture."""

    def __init__(self, config):
        super().__init__()
        self.silence_len = config["glue"]
        self.win = config["win"]
        self.hop = config["hop"]
        self.offset = 0

    def compute(self, enroll, mix):
        """Return the concatenated mono enrollment and mixture waveforms."""

        # Insert the glue signal between enrollment and mixture waveforms.
        if self.silence_len > 0:
            silence = torch.zeros(
                *mix.shape[:-1],
                self.silence_len,
                device=mix.device,
                dtype=mix.dtype,
            )
            prefix = torch.cat([enroll, silence], dim=-1)
        else:
            prefix = enroll

        # Align the mixture onset with an STFT hop boundary.
        prefix_len = prefix.shape[-1]
        pad = (self.hop - ((prefix_len - self.win) % self.hop)) % self.hop
        if pad > 0:
            prefix = torch.cat(
                [
                    prefix,
                    torch.zeros(
                        *mix.shape[:-1],
                        pad,
                        device=mix.device,
                        dtype=mix.dtype,
                    ),
                ],
                dim=-1,
            )

        self.offset = prefix.shape[-1]
        assert self.offset % self.hop == 0
        return torch.cat([prefix, mix], dim=-1)

    def post(self, mix_repr, mix_len=None):
        """Remove the enrollment prefix from a time-domain model output."""
        out = mix_repr[..., self.offset:]
        if mix_len is not None:
            out = out[..., :mix_len]
        return out
