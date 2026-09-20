# Copyright (c) 2024 Changsheng Quan (upstream NBSS implementation)
# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com) (WeSep adaptation)
# SPDX-License-Identifier: MIT
"""NBC2 separator adapted from the official narrow-band implementation.

Reference:
    Changsheng Quan and Xiaofei Li, "NBC2: Multichannel Speech Separation
    with Revised Narrow-band Conformer."
    https://github.com/Audio-WestlakeU/NBSS

WeSep reorganizes the original network into reusable encoder, block,
separator, and decoder modules, and adds waveform and multi-output interfaces.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

from wesep.modules.common.norm import select_norm
from wesep.modules.feature.speech import STFT, iSTFT


class SequenceNorm(nn.Module):
    """Adapt common normalization layers to the current sequence layout."""

    def __init__(
        self,
        norm_type: str,
        dim: int,
        channel_first: bool,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if norm_type == "LN":
            common_type = "cLN" if channel_first else "LN"
            self.transpose = False
        else:
            common_type = norm_type
            self.transpose = not channel_first
        self.norm = select_norm(common_type, dim, group=groups)

    def forward(self, x: Tensor) -> Tensor:
        if self.transpose:
            x = x.transpose(-1, -2)
        x = self.norm(x)
        if self.transpose:
            x = x.transpose(-1, -2)
        return x


class GroupBatchNorm(nn.Module):
    """Share normalization statistics across each sample's frequency bins."""

    def __init__(
        self,
        dim_hidden: int,
        group_size: int,
        share_along_sequence_dim: bool = False,
        transpose: bool = False,
        affine: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dim_hidden = dim_hidden
        self.group_size = group_size
        self.share_along_sequence_dim = share_along_sequence_dim
        self.transpose = transpose
        self.affine = affine
        self.eps = eps

        if affine:
            shape = (dim_hidden, 1) if transpose else (dim_hidden, )
            self.weight = Parameter(torch.ones(shape))
            self.bias = Parameter(torch.zeros(shape))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[0] % self.group_size != 0:
            raise ValueError(
                f"NBC2 grouped batch ({x.shape[0]}) must be divisible by "
                f"frequency bins ({self.group_size})")

        if not self.transpose:
            batch, frames, hidden = x.shape
            grouped = x.reshape(
                batch // self.group_size,
                self.group_size,
                frames,
                hidden,
            )
            dims = (1, 2, 3) if self.share_along_sequence_dim else (1, 3)
        else:
            batch, hidden, frames = x.shape
            grouped = x.reshape(
                batch // self.group_size,
                self.group_size,
                hidden,
                frames,
            )
            dims = (1, 2, 3) if self.share_along_sequence_dim else (1, 2)

        variance, mean = torch.var_mean(grouped,
                                        dim=dims,
                                        unbiased=False,
                                        keepdim=True)
        output = (grouped - mean) / torch.sqrt(variance + self.eps)
        if self.affine:
            output = output * self.weight + self.bias
        return output.reshape_as(x)


class NBC2Encoder(nn.Module):
    """Project each frequency bin independently along the time axis."""

    def __init__(self, input_size: int, dim_hidden: int, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(
            input_size,
            dim_hidden,
            kernel_size,
            padding="same",
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode `[B, input_size, F, T]` into `[B, hidden, F, T]`."""
        batch, channels, frequencies, frames = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch * frequencies, channels,
                                          frames)
        x = self.conv(x)
        return x.reshape(batch, frequencies, -1, frames).permute(0, 2, 1, 3)


class NBC2Decoder(nn.Module):
    """Project narrow-band hidden states to real and imaginary spectra."""

    def __init__(self, dim_hidden: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(dim_hidden, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Decode `[B, hidden, F, T]` into `[B, output_size, F, T]`."""
        x = x.permute(0, 2, 3, 1)
        return self.linear(x).permute(0, 3, 1, 2).contiguous()


class NBC2Block(nn.Module):
    """Model each frequency bin with temporal attention and convolution."""

    def __init__(
        self,
        dim_hidden: int,
        dim_ffn: int,
        n_heads: int,
        dropout: float = 0.0,
        conv_kernel_size: int = 3,
        n_conv_groups: int = 8,
        norms: Tuple[str, str, str] = ("LN", "GBN", "GBN"),
        group_batch_norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        norm_args = group_batch_norm_kwargs or {
            "group_size": 257,
            "share_along_sequence_dim": False,
        }
        self.norm1 = self._new_norm(norms[0], dim_hidden, False, n_conv_groups,
                                    norm_args)
        self.self_attn = nn.MultiheadAttention(dim_hidden,
                                               n_heads,
                                               batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = self._new_norm(norms[1], dim_hidden, False, n_conv_groups,
                                    norm_args)
        self.linear1 = nn.Linear(dim_hidden, dim_ffn)
        self.conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(dim_ffn,
                      dim_ffn,
                      conv_kernel_size,
                      padding="same",
                      groups=n_conv_groups),
            nn.SiLU(),
            nn.Conv1d(dim_ffn,
                      dim_ffn,
                      conv_kernel_size,
                      padding="same",
                      groups=n_conv_groups),
            self._new_norm(norms[2], dim_ffn, True, n_conv_groups, norm_args),
            nn.SiLU(),
            nn.Conv1d(dim_ffn,
                      dim_ffn,
                      conv_kernel_size,
                      padding="same",
                      groups=n_conv_groups),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.linear2 = nn.Linear(dim_ffn, dim_hidden)
        self.dropout2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch, hidden, frequencies, frames = x.shape
        narrow_band = x.permute(0, 2, 3, 1).reshape(batch * frequencies,
                                                    frames, hidden)

        normalized = self.norm1(narrow_band)
        attended, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        narrow_band = narrow_band + self.dropout1(attended)
        normalized = self.norm2(narrow_band)
        convolved = self.linear1(normalized).transpose(-1, -2)
        convolved = self.conv(convolved).transpose(-1, -2)
        narrow_band = narrow_band + self.dropout2(self.linear2(convolved))
        return narrow_band.reshape(batch, frequencies, frames,
                                   hidden).permute(0, 3, 1, 2)

    @staticmethod
    def _new_norm(
        norm_type: str,
        dim_hidden: int,
        transpose: bool,
        num_conv_groups: int,
        group_batch_norm_kwargs: Dict[str, Any],
    ) -> nn.Module:
        if norm_type == "GBN":
            return GroupBatchNorm(
                dim_hidden=dim_hidden,
                transpose=transpose,
                **group_batch_norm_kwargs,
            )
        if norm_type in ("LN", "BN", "GN"):
            return SequenceNorm(
                norm_type,
                dim_hidden,
                channel_first=transpose,
                groups=num_conv_groups,
            )
        raise ValueError(f"Unsupported NBC2 normalization: {norm_type}")


class NBC2Separator(nn.Module):
    """Stack narrow-band temporal modeling blocks."""

    def __init__(
        self,
        n_layers: int,
        dim_hidden: int,
        dim_ffn: int,
        block_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        block_kwargs = block_kwargs or {}
        self.blocks = nn.ModuleList([
            NBC2Block(dim_hidden, dim_ffn, **block_kwargs)
            for _ in range(n_layers)
        ])

    def forward_block(self, x: Tensor, index: int) -> Tensor:
        """Run one public block step for model-level feature insertion."""
        return self.blocks[index](x)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class NBC2(nn.Module):
    """Complete narrow-band complex spectral separator."""

    def __init__(
        self,
        sr: int = 16000,
        win: int = 512,
        stride: int = 256,
        channels: int = 1,
        reference_channel: int = 0,
        nspk: int = 1,
        input_size: Optional[int] = None,
        encoder_kernel_size: int = 5,
        n_layers: int = 8,
        dim_hidden: int = 192,
        dim_ffn: int = 384,
        block_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if not 0 <= reference_channel < channels:
            raise ValueError(
                f"reference_channel must be in [0, {channels}), got "
                f"{reference_channel}")

        self.sr = sr
        self.win = win
        self.stride = stride
        self.channels = channels
        self.reference_channel = reference_channel
        self.nspk = nspk
        self.input_size = 2 * channels if input_size is None else input_size
        self.dim_hidden = dim_hidden
        self.num_layers = n_layers

        # GBN groups the flattened batch by the number of frequency bins.
        block_kwargs = dict(block_kwargs or {})
        group_norm = dict(block_kwargs.get("group_batch_norm_kwargs", {}))
        group_norm["group_size"] = win // 2 + 1
        group_norm.setdefault("share_along_sequence_dim", False)
        block_kwargs["group_batch_norm_kwargs"] = group_norm

        self.stft = STFT(win, stride, win)
        self.encoder = NBC2Encoder(self.input_size, dim_hidden,
                                   encoder_kernel_size)
        self.separator = NBC2Separator(n_layers, dim_hidden, dim_ffn,
                                       block_kwargs)
        self.decoder = NBC2Decoder(dim_hidden, 2 * nspk)
        self.istft = iSTFT(win, stride, win)

    def normalize(self, spectrum: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply per-frequency reference-channel amplitude normalization."""
        reference = spectrum[:, self.reference_channel]
        scale = reference.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        return spectrum / scale.unsqueeze(1), scale

    def decode_spectrum(
        self,
        hidden: Tensor,
        scale: Tensor,
    ) -> Tensor:
        """Decode hidden states into restored `[B, S, F, T]` spectra."""
        spectrum_ri = self.decoder(hidden)
        batch, _, frequencies, frames = spectrum_ri.shape
        spectrum_ri = spectrum_ri.reshape(batch, self.nspk, 2, frequencies,
                                          frames)
        spectrum = torch.complex(spectrum_ri[:, :, 0], spectrum_ri[:, :, 1])
        return spectrum * scale.unsqueeze(1)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.shape[1] != self.channels:
            raise ValueError(f"NBC2 expects [B, {self.channels}, T], got "
                             f"{tuple(x.shape)}")

        # 1. Convert every input channel into the frequency domain.
        spectrum = self.stft(x)[-1]  # [B, C, F, T]

        # 2. Normalize amplitude and concatenate real/imaginary components.
        spectrum, scale = self.normalize(spectrum)
        spectrum_ri = torch.cat(
            [spectrum.real, spectrum.imag],
            dim=1,
        )  # [B, 2*C, F, T]

        # 3. Project each narrow-band input into the hidden dimension.
        hidden = self.encoder(spectrum_ri)  # [B, H, F, T]

        # 4. Model temporal structure independently at every frequency bin.
        hidden = self.separator(hidden)  # [B, H, F, T]

        # 5. Decode and restore the target complex spectra.
        estimated_spectrum = self.decode_spectrum(hidden, scale)  # [B,S,F,T]

        # 6. Convert every estimated source back into a waveform.
        return self.istft(estimated_spectrum, length=x.shape[-1])  # [B,S,L]


if __name__ == "__main__":
    from thop import clever_format, profile

    model = NBC2(
        channels=4,
        nspk=2,
        n_layers=2,
        dim_hidden=32,
        dim_ffn=64,
        block_kwargs={
            "n_heads": 4,
            "n_conv_groups": 8,
            "norms": ("LN", "GBN", "GBN"),
        },
    ).eval()
    mixture = torch.randn(2, 4, 16000)
    with torch.no_grad():
        estimates = model(mixture)
    macs, params = profile(model, inputs=(mixture[:1], ), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"Output shape: {tuple(estimates.shape)}")
    print(f"MACs: {macs}, parameters: {params}")
