# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Reference:
#   J. Han et al., "DPCCN: Densely-Connected Pyramid Complex Convolutional
#   Network for Robust Speech Separation and Extraction," ICASSP 2022.

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.feature.speech import STFT, iSTFT


class CumulativeInstanceNorm(nn.Module):
    """Normalize each channel with statistics from the current and past time."""

    def __init__(self, channels, dimensions, eps=1e-5):
        super().__init__()
        shape = (1, channels) + (1, ) * dimensions
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.eps = eps

    def forward(self, x):
        if x.ndim == 3:
            cumulative_sum = x.cumsum(dim=-1)
            cumulative_square = x.square().cumsum(dim=-1)
            count = torch.arange(1,
                                 x.shape[-1] + 1,
                                 device=x.device,
                                 dtype=x.dtype)
            count = count.view(1, 1, -1)
        elif x.ndim == 4:
            frequency_bins = x.shape[-1]
            cumulative_sum = x.sum(dim=-1, keepdim=True).cumsum(dim=-2)
            cumulative_square = x.square().sum(dim=-1,
                                               keepdim=True).cumsum(dim=-2)
            count = torch.arange(1,
                                 x.shape[-2] + 1,
                                 device=x.device,
                                 dtype=x.dtype)
            count = (count * frequency_bins).view(1, 1, -1, 1)
        else:
            raise ValueError(
                f"CumulativeInstanceNorm expects 3D or 4D input, got "
                f"{tuple(x.shape)}")

        mean = cumulative_sum / count
        variance = cumulative_square / count - mean.square()
        normalized = (x - mean) / torch.sqrt(variance.clamp_min(0) + self.eps)
        return normalized * self.weight + self.bias


class Conv2DBlock(nn.Module):
    """Apply convolution, ELU, and instance normalization."""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            causal=False,
    ):
        super().__init__()
        self.causal = causal
        self.time_padding = padding[0]
        conv_padding = (0, padding[1]) if causal else padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            conv_padding,
        )
        self.activation = nn.ELU()
        self.norm = (CumulativeInstanceNorm(out_channels, dimensions=2)
                     if causal else nn.InstanceNorm2d(out_channels))

    def forward(self, x):
        if self.causal and self.time_padding:
            x = F.pad(x, (0, 0, 2 * self.time_padding, 0))
        return self.norm(self.activation(self.conv(x)))


class ConvTranspose2DBlock(nn.Module):
    """Upsample one encoder scale with optional causal time alignment."""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 2),
            padding=(1, 1),
            output_padding=(0, 0),
            causal=False,
    ):
        super().__init__()
        self.causal = causal
        conv_padding = (0, padding[1]) if causal else padding
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            conv_padding,
            output_padding,
        )
        self.activation = nn.ELU()
        self.norm = (CumulativeInstanceNorm(out_channels, dimensions=2)
                     if causal else nn.InstanceNorm2d(out_channels))

    def forward(self, x, output_size=None):
        x = self.norm(self.activation(self.conv(x)))
        if output_size is not None:
            x = x[..., :output_size[0], :output_size[1]]
        return x


class DenseBlock(nn.Module):
    """Run five densely connected convolutions at one encoder scale."""

    def __init__(self, in_channels, out_channels, mode, causal=False):
        super().__init__()
        if mode not in {"enc", "dec"}:
            raise ValueError("DenseBlock mode must be 'enc' or 'dec'")
        input_scales = 1 if mode == "enc" else 2
        self.layers = nn.ModuleList([
            Conv2DBlock(
                in_channels * (input_scales + index),
                out_channels if index == 4 else in_channels,
                causal=causal,
            ) for index in range(5)
        ])

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return features[-1]


class TCNBlock(nn.Module):
    """Apply one residual dilated temporal convolution."""

    def __init__(
        self,
        channels=384,
        kernel_size=3,
        dilation=1,
        causal=False,
    ):
        super().__init__()
        self.causal = causal
        self.padding = dilation * (kernel_size - 1)
        conv_padding = self.padding if causal else self.padding // 2
        norm = CumulativeInstanceNorm if causal else nn.InstanceNorm1d
        self.norm1 = norm(channels, dimensions=1) if causal else norm(channels)
        self.norm2 = norm(channels, dimensions=1) if causal else norm(channels)
        self.activation1 = nn.ELU()
        self.activation2 = nn.ELU()
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=conv_padding,
            dilation=dilation,
            groups=channels,
        )
        self.output_conv = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        residual = x
        x = self.depthwise_conv(self.activation1(self.norm1(x)))
        if self.causal and self.padding:
            x = x[..., :-self.padding]
        x = self.output_conv(self.activation2(self.norm2(x)))
        return residual + x


class DPCCNEncoder(nn.Module):
    """Encode `[B,16,T,F]` features and retain all U-Net skip tensors."""

    def __init__(self, causal=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                Conv2DBlock(
                    16 if index == 0 else 32,
                    32,
                    stride=(1, 2),
                    causal=causal,
                ),
                DenseBlock(32, 32, "enc", causal=causal),
            ) for index in range(4)
        ])
        self.blocks.extend([
            Conv2DBlock(32, 64, stride=(1, 2), causal=causal),
            Conv2DBlock(64, 128, stride=(1, 2), causal=causal),
            Conv2DBlock(128, 384, stride=(1, 2), causal=causal),
        ])

    def forward_block(self, x, index):
        return self.blocks[index](x)

    def forward(self, x):
        skips = [x]
        for index in range(len(self.blocks)):
            x = self.forward_block(x, index)
            skips.append(x)
        return x, skips


class DPCCNSeparator(nn.Module):
    """Model the flattened time-frequency bottleneck with dilated TCNs."""

    def __init__(self,
                 channels=384,
                 num_layers=2,
                 num_blocks=10,
                 causal=False):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                TCNBlock(channels, dilation=2**index, causal=causal)
                for index in range(num_blocks)
            ]) for _ in range(num_layers)
        ])

    def forward_block(self, x, layer_index, block_index):
        return self.layers[layer_index][block_index](x)

    def forward_layer(self, x, index):
        for block_index in range(len(self.layers[index])):
            x = self.forward_block(x, index, block_index)
        return x

    def forward(self, x):
        batch, channels, frames, frequencies = x.shape
        x = x.reshape(batch, channels, frames * frequencies)
        for index in range(len(self.layers)):
            x = self.forward_layer(x, index)
        return x.reshape(batch, channels, frames, frequencies)


class DPCCNDecoder(nn.Module):
    """Decode the bottleneck with mirrored encoder skip connections."""

    def __init__(self, causal=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvTranspose2DBlock(768, 128, causal=causal),
            ConvTranspose2DBlock(256, 64, causal=causal),
            ConvTranspose2DBlock(128, 32, causal=causal),
        ])
        self.blocks.extend([
            nn.ModuleList([
                DenseBlock(32, 64, "dec", causal=causal),
                ConvTranspose2DBlock(
                    64,
                    16 if index == 3 else 32,
                    causal=causal,
                ),
            ]) for index in range(4)
        ])
        self.final_dense = DenseBlock(16, 32, "dec", causal=causal)

    def forward(self, x, skips):
        reversed_skips = list(reversed(skips))
        for index, block in enumerate(self.blocks):
            skip = reversed_skips[index]
            x = torch.cat([skip, x], dim=1)
            target_size = reversed_skips[index + 1].shape[-2:]
            if isinstance(block, nn.ModuleList):
                x = block[0](x)
                x = block[1](x, output_size=target_size)
            else:
                x = block(x, output_size=target_size)
        return self.final_dense(torch.cat([reversed_skips[-1], x], dim=1))


class PyramidPooling(nn.Module):
    """Aggregate four time-frequency context scales after U-Net decoding."""

    def __init__(self, channels=32, pool_sizes=(4, 8, 16, 32), causal=False):
        super().__init__()
        self.pool_sizes = tuple(pool_sizes)
        self.causal = causal
        self.projections = nn.ModuleList(
            [nn.Conv2d(channels, 8, 1) for _ in self.pool_sizes])

    def forward(self, x):
        time, frequency = x.shape[-2:]
        pooled_features = []
        for size, projection in zip(self.pool_sizes, self.projections):
            time_size = min(size, time)
            frequency_size = min(size, frequency)
            if self.causal:
                padded = F.pad(x, (0, 0, time_size - 1, 0))
                pooled = F.avg_pool2d(
                    padded,
                    kernel_size=(time_size, frequency_size),
                    stride=(1, frequency_size),
                )
            else:
                pooled = F.avg_pool2d(
                    x,
                    kernel_size=(time_size, frequency_size),
                    stride=(time_size, frequency_size),
                )
            pooled = projection(pooled)
            pooled_features.append(
                F.interpolate(
                    pooled,
                    size=(time, frequency),
                    mode="bilinear",
                    align_corners=False,
                ))
        return torch.cat([x, *pooled_features], dim=1)


class DPCCN(nn.Module):
    """Map a mono mixture `[B,1,T]` to `S` separated waveforms `[B,S,T]`."""

    def __init__(
            self,
            win=512,
            stride=128,
            nspk=2,
            tcn_layers=2,
            tcn_blocks=10,
            pool_sizes=(4, 8, 16, 32),
            causal=False,
    ):
        super().__init__()
        if nspk < 1:
            raise ValueError("nspk must be at least 1")

        self.win = win
        self.stride = stride
        self.nspk = nspk
        self.causal = causal
        self.stft = STFT(win, stride, win)
        self.istft = iSTFT(win, stride, win)

        # The public input stage is the speaker fusion point used by TSE models.
        input_padding = (0, 1) if causal else (1, 1)
        self.input_conv = nn.Conv2d(2, 16, 3, padding=input_padding)
        self.input_dense = DenseBlock(16, 16, "enc", causal=causal)
        self.encoder = DPCCNEncoder(causal=causal)
        self.separator = DPCCNSeparator(
            channels=384,
            num_layers=tcn_layers,
            num_blocks=tcn_blocks,
            causal=causal,
        )
        self.decoder = DPCCNDecoder(causal=causal)
        self.pyramid = PyramidPooling(32, pool_sizes, causal=causal)
        self.pyramid_projection = nn.Conv2d(64, 32, 1)
        output_padding = (0, 1) if causal else (1, 1)
        self.output_conv = nn.ConvTranspose2d(32,
                                              2 * nspk,
                                              3,
                                              padding=output_padding)

    def encode_input(self, spec_ri):
        """Run the public pre-encoder stage on `[B,2,T,F]` spectra."""
        if self.causal:
            spec_ri = F.pad(spec_ri, (0, 0, 2, 0))
        return self.input_dense(self.input_conv(spec_ri))

    def reconstruct(self, x, length):
        """Convert `[B,2*S,T,F]` predictions into `[B,S,T]` waveforms."""
        if self.causal:
            x = x[..., :x.shape[-2] - 2, :]
        x = x.permute(0, 1, 3, 2).contiguous()
        batch, _, frequencies, frames = x.shape
        x = x.view(batch, 2, self.nspk, frequencies, frames)
        spectrum = torch.complex(x[:, 0], x[:, 1])
        return self.istft(spectrum, length=length)

    def forward(self, mixture):
        if mixture.ndim == 3 and mixture.shape[1] == 1:
            mixture = mixture.squeeze(1)
        if mixture.ndim != 2:
            raise ValueError(
                f"DPCCN expects [B,T] or [B,1,T], got {tuple(mixture.shape)}")
        length = mixture.shape[-1]

        # 1. Convert the waveform into stacked real and imaginary spectra.
        spectrum = self.stft(mixture)[-1]  # [B, F, T]
        spec_ri = torch.stack(
            [spectrum.real, spectrum.imag],
            dim=1,
        )  # [B, 2, F, T]
        spec_ri = spec_ri.transpose(2, 3)  # [B, 2, T, F]
        # 2. Encode the spectrum and retain one skip at every frequency scale.
        encoded_input = self.encode_input(spec_ri)  # [B, 16, T, F]
        bottleneck, skips = self.encoder(encoded_input)  # [B, 384, T, F_b]
        # 3. Model long-range time-frequency context with dilated TCN layers.
        separated = self.separator(bottleneck)  # [B, 384, T, F_b]
        # 4. Restore the full frequency resolution with U-Net skip connections.
        decoded = self.decoder(separated, skips)  # [B, 32, T, F]
        # 5. Aggregate multi-scale time-frequency context at full resolution.
        decoded = self.pyramid_projection(
            self.pyramid(decoded))  # [B, 32, T, F]
        # 6. Estimate one complex spectrum per source and reconstruct waveforms.
        estimated_spectrum = self.output_conv(decoded)  # [B, 2*S, T, F]
        return self.reconstruct(estimated_spectrum, length)  # [B, S, L]


def check_causal(model, num_samples=16000, change_sample=8000):
    """Report the first output sample affected by future input changes."""
    mixture = torch.randn(1, num_samples)
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
        model = DPCCN(
            nspk=2,
            tcn_layers=1,
            tcn_blocks=3,
            pool_sizes=(2, 4, 8, 16),
            causal=causal_mode,
        ).eval()
        mixture = torch.randn(2, 16003)
        with torch.no_grad():
            estimates = model(mixture)
        macs, params = profile(model, inputs=(mixture[:1], ), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        print(f"causal={causal_mode}, output={tuple(estimates.shape)}")
        print(f"MACs: {macs}, parameters: {params}")
        print(f"first future-affected sample={check_causal(model)}")
