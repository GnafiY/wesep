# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Reference:
#   Y. Luo et al., "Music Source Separation with Band-split RNN"

import numpy as np
import torch
import torch.nn as nn

from wesep.modules.feature.speech import STFT, iSTFT
from wesep.modules.common.norm import select_norm


class BandSplit(nn.Module):

    def __init__(self, band_width):
        super().__init__()
        self.band_width = band_width

    def forward(self, spec):
        """
        spec: (B, F, T)
        spec_RI: (B, 2, F, T) real/imag stacked version
        return:
           subband_spec: list of (B, 2, BW, T)
           subband_mix_spec: list of (B, BW, T) complex
        """
        num_dims = spec.dim()
        assert num_dims == 3 or num_dims == 4, "Only support 3D or 4D Input"

        subband_spec = []
        idx = 0
        for bw in self.band_width:
            if num_dims == 4:
                subband_spec.append(spec[:, :, idx:idx +
                                         bw].contiguous())  # real/imag
            elif num_dims == 3:
                subband_spec.append(spec[:, idx:idx + bw])  # complex
            idx += bw
        return subband_spec


class SubbandNorm(nn.Module):

    def __init__(self,
                 band_width,
                 spec_dim,
                 nband,
                 feature_dim,
                 norm_type='GN',
                 eps=torch.finfo(torch.float32).eps):
        super().__init__()
        self.band_width = band_width
        self.spec_dim = spec_dim
        self.nband = nband
        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    select_norm(norm_type, band_width[i] * spec_dim, eps),
                    nn.Conv1d(band_width[i] * spec_dim, feature_dim, 1),
                ))

    def forward(self, subband_spec):
        """
        subband_spec: list of (B, spec_dim, BW, T)
        return:
            subband_feature: (B, nband, feat, T)
        """
        B = subband_spec[0].shape[0]
        subband_feature = []
        for i, bn in enumerate(self.BN):
            x = subband_spec[i].view(B, self.band_width[i] * self.spec_dim, -1)
            subband_feature.append(bn(x))
        subband_feature = torch.stack(subband_feature, 1)
        return subband_feature


class BandMasker(nn.Module):

    def __init__(self,
                 band_width,
                 nband,
                 feature_dim,
                 norm_type,
                 nspk=1,
                 eps=torch.finfo(torch.float32).eps):
        super().__init__()
        self.band_width = band_width
        self.nspk = nspk
        self.mask = nn.ModuleList([])
        for i in range(nband):
            self.mask.append(
                nn.Sequential(
                    select_norm(norm_type, feature_dim, eps),
                    nn.Conv1d(feature_dim, feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(feature_dim * 4, feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(feature_dim * 4, band_width[i] * 4 * nspk, 1),
                ))

    def forward(self, sep_output, subband_mix_spec):
        """
        sep_output: (B, nband, feat, T)
        subband_mix_spec: list of (B, BW, T) complex
        return:
            est_spec_RI: (B, 2, S, F, T)
        """
        sep_subband_R = []  # real
        sep_subband_I = []  # imag
        for i, mask_func in enumerate(self.mask):
            this_output = mask_func(sep_output[:, i])  # (B*nch, S*4*BW, T)
            Bnch, _, T = this_output.shape
            BW = self.band_width[i]

            # reshape → (B*nch, S, 2, 2, BW, T)
            this_output = this_output.view(Bnch, self.nspk, 2, 2, BW, T)
            # shape→ (B*nch, S, 2, BW, T)
            this_mask = this_output[:, :, 0] * torch.sigmoid(this_output[:, :,
                                                                         1])
            # split real/imag → (B*nch, S, BW, T)
            this_mask_real = this_mask[:, :, 0]
            this_mask_imag = this_mask[:, :, 1]

            # subband_mix_spec[i] shape:
            #   (B*nch, BW, T)
            mix_real = subband_mix_spec[i].real.unsqueeze(
                1)  # → (B*nch,1,BW,T)
            mix_imag = subband_mix_spec[i].imag.unsqueeze(1)

            # apply complex mask:
            est_real = mix_real * this_mask_real - mix_imag * this_mask_imag  # (B*nch,S,BW,T)
            est_imag = mix_real * this_mask_imag + mix_imag * this_mask_real  # (B*nch,S,BW,T)
            sep_subband_R.append(est_real)
            sep_subband_I.append(est_imag)

        # concat on frequency
        est_R = torch.cat(sep_subband_R, dim=2)  # (B*nch, S, F, T)
        est_I = torch.cat(sep_subband_I, dim=2)  # (B*nch, S, F, T)
        est_spec_RI = torch.stack([est_R, est_I], dim=1)  # (B*nch, 2, S, F, T)
        return est_spec_RI


class ResRNN(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 bidirectional=True,
                 norm_type='GN',
                 eps=torch.finfo(torch.float32).eps):
        super(ResRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.eps = eps

        self.norm = select_norm(norm_type, input_size, eps=self.eps)
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # linear projection layer
        self.proj = nn.Linear(hidden_size * (int(bidirectional) + 1),
                              input_size)  # hidden_size = feature_dim * 2

    def forward(self, input, state=None):
        # input shape: batch, dim, seq

        rnn_output, _ = self.rnn(
            self.norm(input).transpose(1, 2).contiguous(),
            state,
        )
        rnn_output = self.proj(rnn_output.contiguous().view(
            -1, rnn_output.shape[2])).view(input.shape[0], input.shape[2],
                                           input.shape[1])

        return input + rnn_output.transpose(1, 2).contiguous()


class BSNet(nn.Module):

    def __init__(self,
                 in_channel,
                 nband=7,
                 causal=False,
                 norm_type='GN',
                 eps=torch.finfo(torch.float32).eps):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband
        self.band_rnn = ResRNN(self.feature_dim,
                               self.feature_dim * 2,
                               bidirectional=not causal,
                               norm_type=norm_type,
                               eps=eps)
        self.band_comm = ResRNN(self.feature_dim,
                                self.feature_dim * 2,
                                bidirectional=True,
                                norm_type='GN',
                                eps=eps)

    def forward(self, input, initial_state=None):
        # input shape: B, nband*N, T
        B, N, T = input.shape

        # Expand sample-level states over the internal band and frame batches.
        band_state = None
        comm_state = None
        if initial_state is not None:
            if "band" in initial_state:
                h0, c0 = initial_state["band"]
                directions = 2 if self.band_rnn.bidirectional else 1
                band_state = (
                    h0.repeat_interleave(self.nband,
                                         dim=0).unsqueeze(0).expand(
                                             directions, -1, -1).contiguous(),
                    c0.repeat_interleave(self.nband,
                                         dim=0).unsqueeze(0).expand(
                                             directions, -1, -1).contiguous(),
                )
            if "communication" in initial_state:
                h0, c0 = initial_state["communication"]
                directions = 2 if self.band_comm.bidirectional else 1
                comm_state = (
                    h0.repeat_interleave(T, dim=0).unsqueeze(0).expand(
                        directions, -1, -1).contiguous(),
                    c0.repeat_interleave(T, dim=0).unsqueeze(0).expand(
                        directions, -1, -1).contiguous(),
                )

        band_output = self.band_rnn(
            input.view(B * self.nband, self.feature_dim, -1),
            band_state).view(B, self.nband, -1, T)

        # band comm
        band_output = (band_output.permute(0, 3, 2, 1).contiguous().view(
            B * T, -1, self.nband))
        output = (self.band_comm(band_output, comm_state).view(
            B, T, -1, self.nband).permute(0, 3, 2, 1).contiguous())

        return output.view(B, N, T)


class BSRNN_Separator(nn.Module):

    def __init__(
        self,
        nband=7,
        num_repeat=6,
        feature_dim=128,
        causal=False,
        norm_type='GN',
        eps=torch.finfo(torch.float32).eps,
    ):
        """
        :param nband : len(self.band_width)
        """
        super(BSRNN_Separator, self).__init__()
        self.nband = nband
        self.feature_dim = feature_dim
        self.separation = nn.ModuleList([])
        for _ in range(num_repeat):
            self.separation.append(
                BSNet(nband * feature_dim, nband, causal, norm_type, eps=eps))

    @property
    def num_blocks(self):
        return len(self.separation)

    def forward_block(self, x, index, initial_state=None):
        """Run one BSNet block on ``[B, nband, feature_dim, T]``."""
        if not 0 <= index < self.num_blocks:
            raise IndexError(f"BSNet block index out of range: {index}")
        batch_size = x.shape[0]
        x = x.view(batch_size, self.nband * self.feature_dim, -1)
        x = self.separation[index](x, initial_state=initial_state)
        return x.view(batch_size, self.nband, self.feature_dim, -1)

    def forward(self, x, initial_states=None):
        """
        x: [B, nband, feature_dim, T]
        initial_states: optional state dict or one state dict per BSNet block
        out: [B, nband, feature_dim, T]
        """
        for idx in range(self.num_blocks):
            state = (initial_states[idx] if isinstance(
                initial_states, (list, tuple)) else initial_states)
            x = self.forward_block(x, idx, initial_state=state)
        return x


class BSRNN(nn.Module):

    def __init__(
        self,
        sr=16000,
        win=512,
        stride=128,
        feature_dim=128,
        num_repeat=6,
        causal=False,
        nspk=1,  # For Separation (multiple output)
        channels=1,
        reference_channel=0,
        spec_dim=None,  # Defaults to real/imag features for all channels.
        eps=torch.finfo(torch.float32).eps,
    ):
        super().__init__()

        self.sr = sr
        self.win = win
        self.stride = stride
        self.feature_dim = feature_dim
        self.nspk = nspk
        self.channels = channels
        if not 0 <= reference_channel < channels:
            raise ValueError(
                f"reference_channel must be in [0, {channels}), got "
                f"{reference_channel}")
        self.reference_channel = reference_channel
        self.spec_dim = 2 * channels if spec_dim is None else spec_dim
        self.eps = eps

        norm_type = "cLN" if causal else "GN"

        # 0-1k (100 hop), 1k-4k (250 hop),
        # 4k-8k (500 hop), 8k-16k (1k hop),
        # 16k-20k (2k hop), 20k-inf
        enc_dim = win // 2 + 1
        # 0-8k (1k hop), 8k-16k (2k hop), 16k
        bandwidth_100 = int(np.floor(100 / (sr / 2.0) * enc_dim))
        bandwidth_200 = int(np.floor(200 / (sr / 2.0) * enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.0) * enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sr / 2.0) * enc_dim))

        # add up to 8k
        band_width = [bandwidth_100] * 15
        band_width += [bandwidth_200] * 10
        band_width += [bandwidth_500] * 5
        band_width += [bandwidth_2k] * 1

        band_width.append(enc_dim - int(np.sum(band_width)))
        nband = len(band_width)
        self.nband = nband

        self.stft = STFT(win, stride, win)
        self.band_split = BandSplit(band_width)
        self.subband_norm = SubbandNorm(
            band_width=band_width,
            spec_dim=self.spec_dim,
            nband=nband,
            feature_dim=feature_dim,
            norm_type=norm_type,
            eps=self.eps,
        )
        self.separator = BSRNN_Separator(
            nband=nband,
            num_repeat=num_repeat,
            feature_dim=feature_dim,
            causal=causal,
            norm_type=norm_type,
            eps=self.eps,
        )
        self.band_masker = BandMasker(
            band_width=band_width,
            nband=nband,
            feature_dim=feature_dim,
            norm_type=norm_type,
            nspk=nspk,
            eps=self.eps,
        )
        self.istft = iSTFT(win, stride, win)

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input):
        # Accept mono [B, T] or configured multichannel audio [B, C, T].
        if input.dim() == 2:
            input = input.unsqueeze(1)
        if input.dim() != 3 or input.shape[1] != self.channels:
            raise ValueError(f"BSRNN expects [B, {self.channels}, T], got "
                             f"{tuple(input.shape)}")

        # 1. Convert into frequency-domain
        spec = self.stft(input)[-1]
        # 2. Concat real and imag, split to subbands
        spec_RI = torch.cat([spec.real, spec.imag], 1)  # (B, 2*C, F, T)
        subband_spec = self.band_split(spec_RI)
        reference_spec = spec[:, self.reference_channel]
        subband_mix_spec = self.band_split(reference_spec)
        # 3. Normalization and bottleneck
        subband_feature = self.subband_norm(
            subband_spec)  # (B, nband, feat, T)
        # 4. Separation
        sep_output = self.separator(subband_feature)  # (B, nband, feat, T)
        # 5. Complex Mask
        est_spec_RI = self.band_masker(sep_output,
                                       subband_mix_spec)  # (B, 2, S, F, T)
        est_complex = torch.complex(est_spec_RI[:, 0],
                                    est_spec_RI[:, 1])  # (B, S, F, T)
        # 6. Back into waveform
        s = self.istft(est_complex)  # (B, S, T)
        return s


def check_causal(model):
    input = torch.randn(1, 16000 * 8).clamp_(-1, 1)
    fs = 16000
    model = model.eval()
    with torch.no_grad():
        out1 = model(input)
        for i in range(fs * 1, fs * 4, fs):
            inputs2 = input.clone()
            inputs2[..., i:] = 1 + torch.rand_like(inputs2[..., i:])
            out2 = model(inputs2)
            print((((out1[0] - out2[0]).abs() > 1e-8).float().argmax()) / fs)
            print((((inputs2 - input).abs() > 1e-8).float().argmax()) / fs)


if __name__ == "__main__":
    from thop import profile, clever_format

    model = BSRNN(
        sr=16000,
        win=512,
        stride=128,
        feature_dim=128,
        num_repeat=6,
        causal=True,
        nspk=2,
    )

    x = torch.randn(4, 32000)
    model = model.eval()
    with torch.no_grad():
        output = model(x)
    macs, params = profile(model, inputs=(x[:1], ), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"MACs: {macs}, parameters: {params}")
    check_causal(model)
