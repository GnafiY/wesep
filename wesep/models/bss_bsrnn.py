# Copyright (c) 2026 Haoyu Li (haoyu.li.cs@smail.nju.edu.cn)
# SPDX-License-Identifier: Apache-2.0
"""BSRNN blind source separation model."""

import torch
import torch.nn as nn

from wesep.modules.separator.bsrnn import BSRNN

class BSS_BSRNN(nn.Module):
    """Extract all speakers from mixture without any condition."""

    def __init__(self, config, defer_pretrained=False):
        super().__init__()
        separator_config = {
            "sr": 16000,
            "win": 512,
            "stride": 128,
            "feature_dim": 128,
            "num_repeat": 6,
            "causal": False,
            "nspk": 2,
            "channels": 1,
            "reference_channel": 0,
        }
        separator_config.update(config.get("separator", {}))
        if separator_config["channels"] != 1:
            raise ValueError(
                "BSS_BSRNN currently supports mono mixtures")
        separator_config["spec_dim"] = 2
        self.sep_model = BSRNN(**separator_config)

    def forward(self, batch):
        """Use `wav_mix` [B,1,T]."""
        mix = batch["wav_mix"]
        if mix.ndim == 2:
            mix = mix.unsqueeze(1)
        if mix.ndim != 3 or mix.shape[1] != 1:
            raise ValueError(
                f"Expected mono mixture [B,1,T], got {tuple(mix.shape)}")

        # S1. Convert the mixture into a complex spectrum.
        mix_spec = self.sep_model.stft(mix)[-1]  # [B, 1, F, T_f]
        spectral_repr = torch.cat([mix_spec.real, mix_spec.imag],
                                  dim=1)  # [B, 2, F, T_f]

        # S2. Split input and reference spectra into frequency bands.
        subband_spec = self.sep_model.band_split(
            spectral_repr)  # N_b * [B, 2, BW, T_f]
        reference_spec = mix_spec[:, 0]  # [B, F, T_f]
        subband_mix_spec = self.sep_model.band_split(
            reference_spec)  # N_b * [B, BW, T_f]

        # S3. Normalize and project every band into the model dimension.
        subband_feature = self.sep_model.subband_norm(
            subband_spec)  # [B, N_b, E, T_f]

        # S4. Separate with the cue-free BSNet blocks.
        sep_output = subband_feature
        for index in range(self.sep_model.separator.num_blocks):
            sep_output = self.sep_model.separator.forward_block(
                sep_output, index)  # [B, N_b, E, T_f]

        # S5. Estimate and apply a complex mask to the mixture spectrum.
        est_spec_ri = self.sep_model.band_masker(
            sep_output, subband_mix_spec)  # [B, 2, S, F, T_f]
        est_spec = torch.complex(est_spec_ri[:, 0],
                                 est_spec_ri[:, 1])  # [B, S, F, T_f]

        # S6. Convert the target spectrum back to its original waveform length.
        speech = self.sep_model.istft(est_spec,
                                      length=mix.shape[-1])  # [B, S, T]
        return {"speech": speech}

def check_causal(model, num_samples=16000):
    """Compare outputs before and after a future mixture perturbation."""
    mixture = torch.randn(1, 1, num_samples).clamp_(-1, 1)
    model = model.eval()
    with torch.no_grad():
        output = model({
            "wav_mix": mixture,
        })["speech"]
        for index in (num_samples // 4, num_samples // 2):
            changed = mixture.clone()
            changed[..., index:] = torch.rand_like(changed[..., index:])
            changed_output = model({
                "wav_mix": changed,
            })["speech"]
            affected = (output -
                        changed_output).abs().gt(1e-8).float().argmax()
            print(f"Output changes at {affected.item() / 16000:.3f} s; "
                  f"input changes at {index / 16000:.3f} s")

if __name__ == "__main__":
    test_config = {
        "separator": {
            "sr": 16000,
            "win": 512,
            "stride": 128,
            "feature_dim": 128,
            "num_repeat": 6,
            "causal": False,
            "nspk": 2,
        },
    }
    model = BSS_BSRNN(test_config).eval()
    test_batch = {
        "wav_mix": torch.randn(2, 1, 3200),
    }
    with torch.no_grad():
        test_output = model(test_batch)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameters / 1e6:.2f} M")
    print({key: tuple(value.shape) for key, value in test_output.items()})
    check_causal(model)
