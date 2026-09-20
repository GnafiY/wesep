# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.common.deep_update import deep_update
from wesep.modules.separator.tfgridnet import TFGridNet
from wesep.modules.speaker.spk_frontend import SpeakerFrontend


class TSE_TFGRIDNET_SPK(nn.Module):
    """Extract target speech with speaker-conditioned TFGridNet."""

    def __init__(self, config, defer_pretrained=False):
        super().__init__()

        separator_config = {
            "nspk": 1,
            "n_fft": 128,
            "stride": 64,
            "channels": 1,
            "n_layers": 6,
            "hidden_channels": 192,
            "n_head": 4,
            "approx_qk_dim": 512,
            "emb_dim": 48,
            "emb_ks": 4,
            "emb_hs": 1,
            "eps": 1e-5,
            "causal": False,
            "estimation": "spec",
            "reference_channel": 0,
            "attention_context": None,
        }
        separator_config.update(config.get("separator", {}))
        n_freqs = separator_config["n_fft"] // 2 + 1
        channels = separator_config["channels"]

        speaker_config = {
            "features": {
                "listen": {
                    "enabled": False,
                    "win": separator_config["n_fft"],
                    "hop": separator_config["stride"],
                },
                "usef": {
                    "enabled": False,
                    "causal": separator_config["causal"],
                    "spec_dim": 2 * channels,
                    "emb_dim": separator_config["emb_dim"],
                    "enc_dim": n_freqs,
                },
                "tfmap": {
                    "enabled": False
                },
                "context": {
                    "enabled": False,
                    "mix_dim": separator_config["emb_dim"],
                    "atten_dim": separator_config["emb_dim"],
                    "band": n_freqs,
                },
                "spkemb": {
                    "enabled": False,
                    "input": "waveform",
                    "mix_dim": separator_config["emb_dim"],
                },
            },
            "speaker_model": {
                "fbank": {
                    "sample_rate": 16000
                },
            },
        }
        self.speaker_config = deep_update(speaker_config,
                                          config.get("speaker", {}))
        features = self.speaker_config["features"]

        # Listen and USEF currently encode mono mixture waveforms.
        for name in ("listen", "usef"):
            if channels > 1 and features[name]["enabled"]:
                raise ValueError(
                    f"{name} currently supports single-channel mixtures only")

        # USEF replaces RI input channels, while TFMap adds one channel.
        input_dim = 2 * channels
        if features["usef"]["enabled"]:
            input_dim = 2 * features["usef"]["emb_dim"]
        if features["tfmap"]["enabled"]:
            input_dim += 1
        separator_config["input_dim"] = input_dim

        self.sep_model = TFGridNet(**separator_config)
        self.spk_ft = SpeakerFrontend(
            self.speaker_config,
            defer_pretrained=defer_pretrained,
        )

    def forward(self, batch):
        """Extract speech from `wav_mix` using the `audio_aux` cue."""
        mixture = batch["wav_mix"]
        enrollment = batch["audio_aux"]
        if mixture.ndim == 2 and self.sep_model.channels == 1:
            mixture = mixture.unsqueeze(1)
        if mixture.ndim != 3 or mixture.shape[1] != self.sep_model.channels:
            raise ValueError(
                f"Expected [B,{self.sep_model.channels},T] mixture, got "
                f"{tuple(mixture.shape)}")
        if self.spk_ft.input_type == "waveform":
            if enrollment.ndim == 3 and enrollment.shape[1] == 1:
                enrollment = enrollment.squeeze(1)
            if enrollment.ndim != 2:
                raise ValueError(
                    "Waveform enrollment must have shape [B,1,T] or [B,T], "
                    f"got {tuple(enrollment.shape)}")

        mix_len = mixture.shape[-1]
        waveform = mixture
        features_config = self.speaker_config["features"]

        # C0. Prepend the enrollment when Listen is enabled.
        if features_config["listen"]["enabled"]:
            waveform = self.spk_ft.listen.compute(
                enrollment,
                waveform.squeeze(1),
            ).unsqueeze(1)  # [B,1,L_p]

        # S1. Normalize and convert the mixture into complex spectra.
        if self.sep_model.causal:
            scale = waveform.new_ones(waveform.shape[0], 1, 1)
        else:
            scale = waveform.std(dim=(1, 2),
                                 keepdim=True).clamp_min(self.sep_model.eps)
        spectrum = self.sep_model.stft(waveform / scale)[-1]  # [B,C,F,T]
        reference_spectrum = spectrum[:, self.sep_model.reference_channel,
                                      ]  # [B, F, T]
        spec_ri = torch.cat(
            [spectrum.real, spectrum.imag],
            dim=1,
        )  # [B, 2*C, F, T]
        enrollment_spectrum = None
        if (features_config["usef"]["enabled"]
                or features_config["tfmap"]["enabled"]):
            enrollment_spectrum = self.sep_model.stft(enrollment)[
                -1]  # [B,F,T_e]

        # C1. Replace RI channels with USEF mixture and enrollment features.
        if features_config["usef"]["enabled"]:
            enrollment_ri = torch.stack(
                [enrollment_spectrum.real, enrollment_spectrum.imag],
                dim=1,
            )  # [B, 2, F, T_e]
            enrollment_usef, mixture_usef = self.spk_ft.usef.compute(
                enrollment_ri,
                spec_ri,
            )  # [B,U,F,T], [B,U,F,T]
            spec_ri = self.spk_ft.usef.post(
                mixture_usef,
                enrollment_usef,
            )  # [B, 2*U, F, T]

        # C2. Append the speaker TF-map to the spectral representation.
        if features_config["tfmap"]["enabled"]:
            tfmap = self.spk_ft.tfmap.compute(
                enrollment_spectrum.abs(),
                reference_spectrum.abs(),
            )  # [B,F,T]
            spec_ri = self.spk_ft.tfmap.post(
                spec_ri,
                tfmap.unsqueeze(1),
            )  # [B, D_in, F, T]

        # S2. Project spectral input into the fixed GridNet width.
        spec_ri = spec_ri.permute(0, 1, 3, 2).contiguous()  # [B,D_in,T,F]
        if self.sep_model.causal:
            spec_ri = F.pad(spec_ri, (0, 0, 2, 0))
        features = self.sep_model.input_norm(
            self.sep_model.input_conv(spec_ri), )  # [B, E, T, F]

        # Speaker fusion uses frequencies as independent feature bands.
        fusion_features = features.permute(
            0,
            3,
            1,
            2,
        ).contiguous()  # [B, F, E, T]

        # C3. Fuse frame-level speaker context after the input projection.
        if features_config["context"]["enabled"]:
            context = self.spk_ft.context.compute(enrollment)  # [B,D_c,T_e]
            fusion_features = self.spk_ft.context.post(
                fusion_features,
                context,
            )  # [B, F, E, T]

        # C4. Fuse one utterance-level speaker embedding.
        if features_config["spkemb"]["enabled"]:
            embedding = self.spk_ft.spkemb.compute(enrollment)  # [B, D]
            if embedding is not None:
                embedding = embedding.unsqueeze(1).unsqueeze(3)
            fusion_features = self.spk_ft.spkemb.post(
                fusion_features,
                embedding,
            )  # [B, F, E, T]
        features = fusion_features.permute(
            0,
            2,
            3,
            1,
        ).contiguous()  # [B, E, T, F]

        # S3. Alternate full-band, sub-band, and cross-frame modeling.
        features = self.sep_model.separator(features)  # [B, E, T, F]

        # S4. Estimate RI spectra for every output source.
        estimated_ri = self.sep_model.output_conv(features)  # [B,2*S,T(+2),F]
        if self.sep_model.causal:
            estimated_ri = estimated_ri[..., :features.shape[-2], :]

        # S5. Interpret every RI pair as a spectrum or complex ratio mask.
        batch_size, _, frames, frequencies = estimated_ri.shape
        estimated_ri = estimated_ri.view(
            batch_size,
            self.sep_model.nspk,
            2,
            frames,
            frequencies,
        )  # [B, S, 2, T, F]
        if self.sep_model.estimation == "mask":
            estimated_ri = 5.0 * torch.tanh(estimated_ri / 5.0)
            complex_mask = torch.complex(estimated_ri[:, :, 0],
                                         estimated_ri[:, :, 1])
            estimated_spectrum = (
                complex_mask *
                reference_spectrum.transpose(-2, -1).unsqueeze(1))
        else:
            estimated_spectrum = torch.complex(estimated_ri[:, :, 0],
                                               estimated_ri[:, :, 1])
        estimated_spectrum = estimated_spectrum.transpose(
            -2,
            -1,
        ).contiguous()  # [B, S, F, T]

        # S6. Reconstruct waveforms and restore the mixture scale.
        speech = self.sep_model.istft(
            estimated_spectrum,
            length=waveform.shape[-1],
        ) * scale  # [B,S,L_p]
        if features_config["listen"]["enabled"]:
            speech = self.spk_ft.listen.post(speech, mix_len=mix_len)
        return {"speech": speech}


def check_causal(model, num_samples=8000, change_sample=4000):
    """Report the first output sample affected by a future mixture change."""
    mixture = torch.randn(1, model.sep_model.channels, num_samples)
    enrollment = torch.randn(1, 1, 3200)
    changed = mixture.clone()
    changed[..., change_sample:] = torch.randn_like(changed[...,
                                                            change_sample:])
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
    config = {
        "separator": {
            "n_layers": 2,
            "hidden_channels": 32,
            "emb_dim": 16,
            "n_head": 4,
            "causal": True,
            "estimation": "mask",
            "attention_context": 32,
        },
        "speaker": {
            "features": {
                "listen": {
                    "enabled": True,
                    "glue": 128
                },
                "usef": {
                    "enabled": True
                },
                "tfmap": {
                    "enabled": True
                },
                "context": {
                    "enabled": True
                },
                "spkemb": {
                    "enabled": True
                },
            },
        },
    }
    model = TSE_TFGRIDNET_SPK(config, defer_pretrained=True).eval()
    inputs = {
        "wav_mix": torch.randn(2, 1, 3200),
        "audio_aux": torch.randn(2, 1, 1600),
    }
    with torch.no_grad():
        output = model(inputs)
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"output={tuple(output['speech'].shape)}")
    print(f"first future-affected sample={check_causal(model)}")
