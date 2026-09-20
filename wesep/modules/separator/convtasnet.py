# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Reference:
#   Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency
#   Magnitude Masking for Speech Separation."

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.common.norm import select_norm


class ConvTasNetEncoder(nn.Module):
    """Encode `[B, C, T]` waveforms into `[B, N, K]` features."""

    def __init__(self, channels, enc_dim, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            enc_dim,
            kernel_size,
            stride=stride,
            bias=False,
        )

    def forward(self, x):
        return F.relu(self.conv(x))


class ConvTasNetDecoder(nn.Module):
    """Decode masked features `[B, S, N, K]` into `[B, S, T]`."""

    def __init__(self, enc_dim, kernel_size, stride):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            enc_dim,
            1,
            kernel_size,
            stride=stride,
            bias=False,
        )

    def forward(self, x, length=None):
        batch, sources, features, frames = x.shape
        x = x.reshape(batch * sources, features, frames)
        x = self.deconv(x).reshape(batch, sources, -1)
        return x[..., :length] if length is not None else x


class TCNBlock(nn.Module):
    """Apply one residual depthwise convolutional TCN block."""

    def __init__(
        self,
        bottleneck_dim,
        conv_dim,
        kernel_size,
        dilation,
        norm_type,
        causal,
        skip_connection,
    ):
        super().__init__()
        if not causal and kernel_size % 2 == 0:
            raise ValueError("Noncausal TCN blocks require an odd kernel size")

        self.causal = causal
        self.padding = dilation * (kernel_size - 1)
        conv_padding = self.padding if causal else self.padding // 2

        self.input_proj = nn.Conv1d(bottleneck_dim, conv_dim, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm_type, conv_dim)
        self.depthwise_conv = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size,
            padding=conv_padding,
            dilation=dilation,
            groups=conv_dim,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm_type, conv_dim)
        self.residual_proj = nn.Conv1d(conv_dim, bottleneck_dim, 1)
        self.skip_proj = (nn.Conv1d(conv_dim, bottleneck_dim, 1)
                          if skip_connection else None)

    def forward(self, x):
        residual = x
        x = self.norm1(self.prelu1(self.input_proj(x)))
        x = self.depthwise_conv(x)
        if self.causal and self.padding:
            x = x[..., :-self.padding]
        x = self.norm2(self.prelu2(x))
        skip = self.skip_proj(x) if self.skip_proj is not None else None
        return residual + self.residual_proj(x), skip


class TCNRepeat(nn.Module):
    """Run one dilation cycle and return its residual and skip outputs."""

    def __init__(
        self,
        num_blocks,
        bottleneck_dim,
        conv_dim,
        kernel_size,
        dilation_start,
        norm_type,
        causal,
        skip_connection,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TCNBlock(
                bottleneck_dim=bottleneck_dim,
                conv_dim=conv_dim,
                kernel_size=kernel_size,
                dilation=2**(index + dilation_start),
                norm_type=norm_type,
                causal=causal,
                skip_connection=skip_connection,
            ) for index in range(num_blocks)
        ])

    @property
    def num_blocks(self):
        return len(self.blocks)

    def forward_block(self, x, index):
        if not 0 <= index < self.num_blocks:
            raise IndexError(f"TCN block index out of range: {index}")
        return self.blocks[index](x)

    def forward(self, x):
        skip_sum = None
        for index in range(self.num_blocks):
            x, skip = self.forward_block(x, index)
            if skip is not None:
                skip_sum = skip if skip_sum is None else skip_sum + skip
        return x, skip_sum


class TCNSeparator(nn.Module):
    """Separate bottleneck features with repeated dilation cycles."""

    def __init__(
        self,
        num_repeats,
        num_blocks,
        bottleneck_dim,
        conv_dim,
        kernel_size,
        dilation_start=0,
        causal=False,
        skip_connection=True,
    ):
        super().__init__()
        norm_type = "cLN" if causal else "gLN"
        self.repeats = nn.ModuleList([
            TCNRepeat(
                num_blocks=num_blocks,
                bottleneck_dim=bottleneck_dim,
                conv_dim=conv_dim,
                kernel_size=kernel_size,
                dilation_start=dilation_start,
                norm_type=norm_type,
                causal=causal,
                skip_connection=skip_connection,
            ) for _ in range(num_repeats)
        ])

    @property
    def num_repeats(self):
        return len(self.repeats)

    @property
    def num_blocks(self):
        return self.repeats[0].num_blocks

    def forward_block(self, x, repeat_index, block_index):
        if not 0 <= repeat_index < self.num_repeats:
            raise IndexError(f"TCN repeat index out of range: {repeat_index}")
        return self.repeats[repeat_index].forward_block(x, block_index)

    def forward_repeat(self, x, index):
        if not 0 <= index < self.num_repeats:
            raise IndexError(f"TCN repeat index out of range: {index}")
        return self.repeats[index](x)

    def forward(self, x):
        skip_sum = None
        for index in range(self.num_repeats):
            x, repeat_skip = self.forward_repeat(x, index)
            if repeat_skip is not None:
                skip_sum = (repeat_skip if skip_sum is None else skip_sum +
                            repeat_skip)
        return skip_sum if skip_sum is not None else x


class MaskGenerator(nn.Module):
    """Generate `S` masks shaped `[B, S, N, K]`."""

    def __init__(self, bottleneck_dim, enc_dim, nspk, activation="relu"):
        super().__init__()
        if activation not in {"relu", "sigmoid", "softmax"}:
            raise ValueError(f"Unsupported mask activation: {activation}")
        self.enc_dim = enc_dim
        self.nspk = nspk
        self.activation = activation
        self.prelu = nn.PReLU()
        self.proj = nn.Conv1d(bottleneck_dim, nspk * enc_dim, 1)

    def forward(self, x):
        batch, _, frames = x.shape
        masks = self.proj(self.prelu(x)).view(batch, self.nspk, self.enc_dim,
                                              frames)
        if self.activation == "relu":
            return F.relu(masks)
        if self.activation == "sigmoid":
            return torch.sigmoid(masks)
        return torch.softmax(masks, dim=1)


class ConvTasNet(nn.Module):
    """Time-domain separator mapping `[B, 1, T]` to `[B, S, T]`."""

    def __init__(
        self,
        enc_dim=512,
        kernel_size=16,
        bottleneck_dim=128,
        conv_dim=512,
        conv_kernel_size=3,
        num_blocks=8,
        num_repeats=3,
        nspk=2,
        causal=False,
        skip_connection=True,
        mask_activation="relu",
    ):
        super().__init__()
        if kernel_size < 2 or kernel_size % 2:
            raise ValueError(
                "Encoder kernel_size must be an even integer >= 2")
        if nspk < 1:
            raise ValueError("nspk must be at least 1")

        self.nspk = nspk
        self.kernel_size = kernel_size
        self.stride = kernel_size // 2

        self.encoder = ConvTasNetEncoder(
            channels=1,
            enc_dim=enc_dim,
            kernel_size=kernel_size,
            stride=self.stride,
        )
        norm_type = "cLN" if causal else "gLN"
        self.input_norm = select_norm(norm_type, enc_dim)
        self.bottleneck = nn.Conv1d(enc_dim, bottleneck_dim, 1)
        self.separator = TCNSeparator(
            num_repeats=num_repeats,
            num_blocks=num_blocks,
            bottleneck_dim=bottleneck_dim,
            conv_dim=conv_dim,
            kernel_size=conv_kernel_size,
            causal=causal,
            skip_connection=skip_connection,
        )
        self.masker = MaskGenerator(
            bottleneck_dim=bottleneck_dim,
            enc_dim=enc_dim,
            nspk=nspk,
            activation=mask_activation,
        )
        self.decoder = ConvTasNetDecoder(
            enc_dim=enc_dim,
            kernel_size=kernel_size,
            stride=self.stride,
        )

    def pad_input(self, x):
        """Right-pad `[B, 1, T]` for exact encoder-decoder reconstruction."""
        length = x.shape[-1]
        padding = max(self.kernel_size - length, 0)
        padded_length = length + padding
        padding += (
            self.stride -
            (padded_length - self.kernel_size) % self.stride) % self.stride
        return F.pad(x, (0, padding)), length

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(
                f"ConvTasNet expects [B, 1, T] or [B, T], got {tuple(x.shape)}"
            )

        x, length = self.pad_input(x)

        # 1. Encode the mixture waveform.
        encoded = self.encoder(x)  # [B, N, K]
        # 2. Normalize and project into the TCN bottleneck.
        bottleneck = self.bottleneck(self.input_norm(encoded))  # [B, H, K]
        # 3. Separate the bottleneck features with repeated TCN blocks.
        separated = self.separator(bottleneck)  # [B, H, K]
        # 4. Estimate one encoder mask for each output source.
        masks = self.masker(separated)  # [B, S, N, K]
        # 5. Apply the masks to the shared mixture representation.
        masked = masks * encoded.unsqueeze(1)  # [B, S, N, K]
        # 6. Decode each source and restore the original waveform length.
        return self.decoder(masked, length=length)  # [B, S, L]


def check_causal(model):
    """Report the first output sample affected by future input changes."""
    mixture = torch.randn(1, 16000)
    changed = mixture.clone()
    changed[..., 8000:] = torch.randn_like(changed[..., 8000:])
    model = model.eval()
    with torch.no_grad():
        output = model(mixture)
        changed_output = model(changed)
    difference = (output - changed_output).abs().amax(dim=1).squeeze(0)
    affected = torch.nonzero(difference > 1e-6)
    return affected[0].item() if affected.numel() else None


if __name__ == "__main__":
    from thop import clever_format, profile

    model = ConvTasNet(
        enc_dim=64,
        bottleneck_dim=32,
        conv_dim=64,
        num_blocks=4,
        num_repeats=2,
        nspk=2,
        causal=True,
    ).eval()
    mixture = torch.randn(2, 16000)
    with torch.no_grad():
        estimates = model(mixture)
    macs, params = profile(model, inputs=(mixture[:1], ), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"Output shape: {tuple(estimates.shape)}")
    print(f"MACs: {macs}, parameters: {params}")
    print(f"First future-affected sample: {check_causal(model)}")
