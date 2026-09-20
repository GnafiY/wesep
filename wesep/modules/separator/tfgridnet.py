# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# This implementation is based on ESPnet's Apache-2.0 TFGridNetV2:
# https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/
# tfgridnetv2_separator.py
#
# Reference:
#   Z.-Q. Wang et al., "TF-GridNet: Integrating Full- and Sub-Band Modeling
#   for Speech Separation," IEEE/ACM TASLP, 2023.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.feature.speech import STFT, iSTFT


class LayerNormalization4DCF(nn.Module):
    """Normalize channel-frequency features independently at every frame."""

    def __init__(self, channels, frequencies, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, frequencies))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, frequencies))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(1, 3), keepdim=True)
        variance = x.var(dim=(1, 3), unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(variance +
                                       self.eps) * self.weight + self.bias


class LayerNormalization4DC(nn.Module):
    """Normalize channels independently at every time-frequency bin."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(variance +
                                       self.eps) * self.weight + self.bias


class AllHeadPReLULayerNormalization4DCF(nn.Module):
    """Apply per-head PReLU and frame-wise channel-frequency normalization."""

    def __init__(self, heads, channels, frequencies, eps=1e-5):
        super().__init__()
        self.heads = heads
        self.channels = channels
        self.frequencies = frequencies
        self.weight = nn.Parameter(
            torch.ones(1, heads, channels, 1, frequencies))
        self.bias = nn.Parameter(
            torch.zeros(1, heads, channels, 1, frequencies))
        self.activation = nn.PReLU(num_parameters=heads, init=0.25)
        self.eps = eps

    def forward(self, x):
        batch, _, frames, frequencies = x.shape
        if frequencies != self.frequencies:
            raise ValueError(
                f"Expected {self.frequencies} frequency bins, got "
                f"{frequencies}")
        x = x.view(batch, self.heads, self.channels, frames, frequencies)
        x = self.activation(x)
        mean = x.mean(dim=(2, 4), keepdim=True)
        variance = x.var(dim=(2, 4), unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(variance +
                                       self.eps) * self.weight + self.bias


class GridNetBlock(nn.Module):
    """Run full-band, sub-band, and cross-frame modeling on `[B,C,T,F]`."""

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        approx_qk_dim=512,
        eps=1e-5,
        causal=False,
        attention_context=None,
        attention_output_norm="channel_frequency",
    ):
        super().__init__()
        if emb_hs < 1 or emb_ks < emb_hs:
            raise ValueError("Require emb_ks >= emb_hs >= 1")
        if causal and emb_hs != 1:
            raise ValueError("Causal TFGridNet currently requires emb_hs=1")
        if emb_dim % n_head:
            raise ValueError("emb_dim must be divisible by n_head")
        if attention_context is not None and attention_context < 1:
            raise ValueError("attention_context must be positive or None")
        if attention_context is not None and not causal:
            raise ValueError("attention_context is only valid in causal mode")
        if attention_output_norm not in ("channel", "channel_frequency"):
            raise ValueError("attention_output_norm must be 'channel' or "
                             "'channel_frequency'")

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head
        self.causal = causal
        self.attention_context = attention_context
        in_channels = emb_dim * emb_ks

        # Intra-frame modeling runs over frequency and remains bidirectional.
        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels,
            hidden_channels,
            batch_first=True,
            bidirectional=True,
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(2 * hidden_channels, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                2 * hidden_channels,
                emb_dim,
                emb_ks,
                stride=emb_hs,
            )

        # Inter-frame modeling is unidirectional only in causal mode.
        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels,
            hidden_channels,
            batch_first=True,
            bidirectional=not causal,
        )
        inter_channels = hidden_channels if causal else 2 * hidden_channels
        if causal or emb_ks != emb_hs:
            self.inter_linear = nn.ConvTranspose1d(
                inter_channels,
                emb_dim,
                emb_ks,
                stride=emb_hs,
            )
        else:
            self.inter_linear = nn.Linear(inter_channels, in_channels)

        attention_dim = math.ceil(approx_qk_dim / n_freqs)
        value_dim = emb_dim // n_head
        self.attn_conv_q = nn.Conv2d(emb_dim, n_head * attention_dim, 1)
        self.attn_conv_k = nn.Conv2d(emb_dim, n_head * attention_dim, 1)
        self.attn_conv_v = nn.Conv2d(emb_dim, n_head * value_dim, 1)
        self.attn_norm_q = AllHeadPReLULayerNormalization4DCF(
            n_head, attention_dim, n_freqs, eps)
        self.attn_norm_k = AllHeadPReLULayerNormalization4DCF(
            n_head, attention_dim, n_freqs, eps)
        self.attn_norm_v = AllHeadPReLULayerNormalization4DCF(
            n_head, value_dim, n_freqs, eps)
        output_norm = (LayerNormalization4DC(emb_dim, eps)
                       if attention_output_norm == "channel" else
                       LayerNormalization4DCF(emb_dim, n_freqs, eps))
        self.attn_projection = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            nn.PReLU(),
            output_norm,
        )

    def forward(self, x):
        batch, channels, old_frames, old_freqs = x.shape
        overlap = self.emb_ks - self.emb_hs
        padded_freqs = (math.ceil(
            (old_freqs + 2 * overlap - self.emb_ks) / self.emb_hs) *
                        self.emb_hs + self.emb_ks)
        if self.causal:
            padded_frames = old_frames
            x = x.permute(0, 2, 3, 1)
            x = F.pad(
                x,
                (0, 0, overlap, padded_freqs - old_freqs - overlap),
            )
        else:
            padded_frames = (math.ceil(
                (old_frames + 2 * overlap - self.emb_ks) / self.emb_hs) *
                             self.emb_hs + self.emb_ks)
            x = x.permute(0, 2, 3, 1)
            x = F.pad(
                x,
                (
                    0,
                    0,
                    overlap,
                    padded_freqs - old_freqs - overlap,
                    overlap,
                    padded_frames - old_frames - overlap,
                ),
            )

        # 1. Model frequency patches independently in every frame.
        residual = x
        intra = self.intra_norm(x)  # [B, T, F_p, E]
        if self.emb_ks == self.emb_hs:
            intra = intra.view(
                batch * padded_frames,
                -1,
                self.emb_ks * channels,
            )
            intra, _ = self.intra_rnn(intra)
            intra = self.intra_linear(intra)
            intra = intra.view(batch, padded_frames, padded_freqs, channels)
        else:
            intra = intra.view(batch * padded_frames, padded_freqs, channels)
            intra = intra.transpose(1, 2)
            intra = F.unfold(
                intra[..., None],
                (self.emb_ks, 1),
                stride=(self.emb_hs, 1),
            ).transpose(1, 2)
            intra, _ = self.intra_rnn(intra)
            intra = self.intra_linear(intra.transpose(1, 2))
            intra = intra.view(batch, padded_frames, channels, padded_freqs)
            intra = intra.transpose(-2, -1)
        intra = intra + residual  # [B, T, F_p, E]

        # 2. Model each sub-band over time.
        inter = intra.transpose(1, 2)  # [B, F_p, T, E]
        residual = inter
        inter = self.inter_norm(inter)
        if self.causal:
            inter = inter.reshape(batch * padded_freqs, old_frames, channels)
            inter = inter.transpose(1, 2)
            inter = F.pad(inter, (self.emb_ks - 1, 0))
            inter = F.unfold(
                inter[..., None],
                (self.emb_ks, 1),
                stride=(1, 1),
            ).transpose(1, 2)
            inter, _ = self.inter_rnn(inter)
            inter = self.inter_linear(inter.transpose(1, 2))
            inter = inter[..., :old_frames]
            inter = inter.view(batch, padded_freqs, channels, old_frames)
            inter = inter.transpose(-2, -1)
        elif self.emb_ks == self.emb_hs:
            inter = inter.view(
                batch * padded_freqs,
                -1,
                self.emb_ks * channels,
            )
            inter, _ = self.inter_rnn(inter)
            inter = self.inter_linear(inter)
            inter = inter.view(batch, padded_freqs, padded_frames, channels)
        else:
            inter = inter.view(batch * padded_freqs, padded_frames, channels)
            inter = inter.transpose(1, 2)
            inter = F.unfold(
                inter[..., None],
                (self.emb_ks, 1),
                stride=(self.emb_hs, 1),
            ).transpose(1, 2)
            inter, _ = self.inter_rnn(inter)
            inter = self.inter_linear(inter.transpose(1, 2))
            inter = inter.view(batch, padded_freqs, channels, padded_frames)
            inter = inter.transpose(-2, -1)
        inter = inter + residual
        inter = inter.permute(0, 3, 2, 1)  # [B, E, T_p, F_p]
        if self.causal:
            inter = inter[..., :old_frames, overlap:overlap + old_freqs]
        else:
            inter = inter[..., overlap:overlap + old_frames,
                          overlap:overlap + old_freqs, ]

        # 3. Attend across frames while preserving the full frequency axis.
        query = self.attn_norm_q(self.attn_conv_q(inter))
        key = self.attn_norm_k(self.attn_conv_k(inter))
        value = self.attn_norm_v(self.attn_conv_v(inter))
        query = query.view(-1, *query.shape[2:])
        key = key.view(-1, *key.shape[2:])
        value = value.view(-1, *value.shape[2:])
        query = query.transpose(1, 2).flatten(start_dim=2)
        key = key.transpose(2, 3).reshape(batch * self.n_head, -1, old_frames)
        value = value.transpose(1, 2)
        value_shape = value.shape
        value = value.flatten(start_dim=2)
        attention = torch.matmul(query, key) / math.sqrt(query.shape[-1],
                                                         )  # [B*H, T, T]
        if self.causal:
            positions = torch.arange(old_frames, device=x.device)
            query_positions = positions[:, None]
            key_positions = positions[None, :]
            blocked = key_positions > query_positions
            if self.attention_context is not None:
                blocked |= key_positions < (query_positions -
                                            self.attention_context + 1)
            attention = attention.masked_fill(blocked, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        value = torch.matmul(attention, value).reshape(value_shape)
        value = value.transpose(1, 2)
        value = value.reshape(batch, self.emb_dim, old_frames, old_freqs)
        return self.attn_projection(value) + inter  # [B, E, T, F]


class TFGridNetSeparator(nn.Module):
    """Apply a configurable sequence of public GridNet blocks."""

    def __init__(
        self,
        n_layers,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head,
        approx_qk_dim,
        eps,
        causal,
        attention_context,
        attention_output_norm,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            GridNetBlock(
                emb_dim=emb_dim,
                emb_ks=emb_ks,
                emb_hs=emb_hs,
                n_freqs=n_freqs,
                hidden_channels=hidden_channels,
                n_head=n_head,
                approx_qk_dim=approx_qk_dim,
                eps=eps,
                causal=causal,
                attention_context=attention_context,
                attention_output_norm=attention_output_norm,
            ) for _ in range(n_layers)
        ])

    @property
    def num_blocks(self):
        return len(self.blocks)

    def forward_block(self, x, index):
        if not 0 <= index < self.num_blocks:
            raise IndexError(f"GridNet block index out of range: {index}")
        return self.blocks[index](x)

    def forward(self, x):
        for index in range(self.num_blocks):
            x = self.forward_block(x, index)
        return x


class TFGridNet(nn.Module):
    """Map `[B,C,T]` mixtures to `S` separated waveforms `[B,S,T]`.

    Causal mode prevents future-frame access inside GridNet. The centered STFT
    still introduces a fixed analysis-window lookahead.
    """

    def __init__(
        self,
        nspk=2,
        n_fft=128,
        stride=64,
        channels=1,
        n_layers=6,
        hidden_channels=192,
        n_head=4,
        approx_qk_dim=512,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        eps=1e-5,
        causal=False,
        input_dim=None,
        estimation="spec",
        reference_channel=0,
        attention_context=None,
        attention_output_norm="channel_frequency",
    ):
        super().__init__()
        if nspk < 1:
            raise ValueError("nspk must be at least 1")
        if channels < 1:
            raise ValueError("channels must be at least 1")
        if n_fft % 2:
            raise ValueError("n_fft must be even")
        if estimation not in ("spec", "mask"):
            raise ValueError("estimation must be 'spec' or 'mask'")
        if not 0 <= reference_channel < channels:
            raise ValueError(
                f"reference_channel must be in [0, {channels}), got "
                f"{reference_channel}")

        self.nspk = nspk
        self.channels = channels
        self.n_fft = n_fft
        self.stride = stride
        self.emb_dim = emb_dim
        self.causal = causal
        self.attention_context = attention_context
        self.eps = eps
        self.estimation = estimation
        self.reference_channel = reference_channel
        n_freqs = n_fft // 2 + 1
        self.n_freqs = n_freqs
        self.input_dim = 2 * channels if input_dim is None else input_dim
        self.stft = STFT(n_fft, stride, n_fft)
        self.istft = iSTFT(n_fft, stride, n_fft)

        input_padding = (0, 1) if causal else (1, 1)
        self.input_conv = nn.Conv2d(
            self.input_dim,
            emb_dim,
            kernel_size=(3, 3),
            padding=input_padding,
        )
        self.input_norm = (LayerNormalization4DCF(emb_dim, n_freqs, eps)
                           if causal else nn.GroupNorm(1, emb_dim, eps=eps))
        self.separator = TFGridNetSeparator(
            n_layers=n_layers,
            emb_dim=emb_dim,
            emb_ks=emb_ks,
            emb_hs=emb_hs,
            n_freqs=n_freqs,
            hidden_channels=hidden_channels,
            n_head=n_head,
            approx_qk_dim=approx_qk_dim,
            eps=eps,
            causal=causal,
            attention_context=attention_context,
            attention_output_norm=attention_output_norm,
        )
        output_padding = (0, 1) if causal else (1, 1)
        self.output_conv = nn.ConvTranspose2d(
            emb_dim,
            2 * nspk,
            kernel_size=(3, 3),
            padding=output_padding,
        )

    def forward(self, mixture):
        """Separate `[B,C,T]` mixtures into `[B,S,T]` waveforms."""
        if mixture.ndim == 2:
            mixture = mixture.unsqueeze(1)
        if mixture.ndim != 3 or mixture.shape[1] != self.channels:
            raise ValueError(f"TFGridNet expects [B,{self.channels},T], got "
                             f"{tuple(mixture.shape)}")
        length = mixture.shape[-1]

        # S1. Normalize and convert the mixture into complex spectra.
        if self.causal:
            scale = mixture.new_ones(mixture.shape[0], 1, 1)
        else:
            scale = mixture.std(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        spectrum = self.stft(mixture / scale)[-1]  # [B, C, F, T]

        # S2. Stack RI components and project them into GridNet features.
        spec_ri = torch.cat(
            [spectrum.real, spectrum.imag],
            dim=1,
        )  # [B, 2*C, F, T]
        spec_ri = spec_ri.permute(0, 1, 3, 2).contiguous()  # [B, 2*C, T, F]
        if self.causal:
            spec_ri = F.pad(spec_ri, (0, 0, 2, 0))
        features = self.input_norm(self.input_conv(spec_ri))  # [B, E, T, F]

        # S3. Alternate full-band, sub-band, and cross-frame modeling.
        features = self.separator(features)  # [B, E, T, F]

        # S4. Estimate RI spectra for every output source.
        estimated_ri = self.output_conv(features)  # [B, 2*S, T(+2), F]
        if self.causal:
            estimated_ri = estimated_ri[..., :features.shape[-2], :]

        # S5. Interpret each RI pair as a spectrum or complex ratio mask.
        batch, _, frames, frequencies = estimated_ri.shape
        estimated_ri = estimated_ri.view(
            batch,
            self.nspk,
            2,
            frames,
            frequencies,
        )  # [B, S, 2, T, F]
        if self.estimation == "mask":
            # Smoothly compress cRM components to the paper's [-5, 5] range.
            estimated_ri = 5.0 * torch.tanh(estimated_ri / 5.0)
            complex_mask = torch.complex(estimated_ri[:, :, 0],
                                         estimated_ri[:, :, 1])
            reference_spectrum = spectrum[:, self.reference_channel].transpose(
                -2, -1)
            estimated_spectrum = complex_mask * reference_spectrum.unsqueeze(1)
        else:
            estimated_spectrum = torch.complex(estimated_ri[:, :, 0],
                                               estimated_ri[:, :, 1])
        estimated_spectrum = estimated_spectrum.transpose(
            -2,
            -1,
        ).contiguous()  # [B, S, F, T]

        # S6. Reconstruct waveforms and restore the mixture scale.
        return self.istft(estimated_spectrum, length=length) * scale  # [B,S,L]


def check_causal(model, num_samples=8000, change_sample=4000):
    """Report the first output sample affected by a future input change."""
    mixture = torch.randn(1, model.channels, num_samples)
    changed = mixture.clone()
    changed[..., change_sample:] = torch.randn_like(changed[...,
                                                            change_sample:])
    model = model.eval()
    with torch.no_grad():
        output = model(mixture)
        changed_output = model(changed)
    difference = (output - changed_output).abs().amax(dim=1).squeeze(0)
    affected = torch.nonzero(difference > 1e-6)
    return affected[0].item() if affected.numel() else None


if __name__ == "__main__":
    from thop import clever_format, profile

    for causal_mode in (False, True):
        model = TFGridNet(
            nspk=2,
            n_layers=2,
            hidden_channels=32,
            emb_dim=16,
            n_head=4,
            causal=causal_mode,
        ).eval()
        mixture = torch.randn(2, 1, 3203)
        with torch.no_grad():
            estimates = model(mixture)
        macs, params = profile(model, inputs=(mixture[:1], ), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        print(f"causal={causal_mode}, output={tuple(estimates.shape)}")
        print(f"MACs: {macs}, parameters: {params}")
        print(f"first future-affected sample={check_causal(model)}")
