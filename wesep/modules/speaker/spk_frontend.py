# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Description: Speaker feature assembly for WeSep models.

import torch.nn as nn

from wesep.modules.common.deep_update import deep_update
from wesep.modules.fusion.speech import SpeakerFuseLayer
from wesep.modules.speaker.encoder import Fbank_kaldi, SpeakerEncoder
from wesep.modules.speaker.listen import ListenFeature
from wesep.modules.speaker.multi_level import ContextFeature, TFMapFeature
from wesep.modules.speaker.spex_plus import SpExPlusFeature
from wesep.modules.speaker.usef import UsefFeature


class SpeakerEmbFeature(nn.Module):
    """Extract and fuse an utterance-level speaker embedding."""

    def __init__(
        self,
        conf_emb,
        conf_spk,
        fbank=None,
        encoder=None,
        defer_pretrained=False,
    ):
        super().__init__()
        self.input_type = conf_emb["input"]
        speaker_model = conf_emb["speaker_model"] or conf_spk
        encoder_conf = speaker_model["speaker_encoder"]
        self.embed_dim = encoder_conf["spk_args"]["embed_dim"]

        if self.input_type == "waveform" and conf_emb["speaker_model"]:
            self.fbank = Fbank_kaldi(**speaker_model["fbank"])
            self.encoder = SpeakerEncoder(
                encoder_conf,
                defer_pretrained=defer_pretrained,
            )
        elif self.input_type == "waveform":
            self.fbank = fbank
            self.encoder = encoder

        self.fusionLayer = SpeakerFuseLayer(
            embed_dim=self.embed_dim,
            feat_dim=conf_emb["mix_dim"],
            fuse_type=conf_emb["fusion"],
        )

    def compute(self, enroll, mix=None, present=None):
        """Extract an utterance-level speaker embedding ``(B,F)``."""
        if present is not None and not present.any():
            return None

        valid_enroll = enroll if present is None else enroll[present]
        if self.input_type == "embedding":
            if enroll.ndim != 2 or enroll.shape[-1] != self.embed_dim:
                raise ValueError(
                    "Precomputed speaker embeddings must have shape "
                    f"[B, {self.embed_dim}], got {tuple(enroll.shape)}")
            emb = valid_enroll
        else:
            fb = self.fbank(valid_enroll)
            emb = self.encoder(fb)
            if isinstance(emb, tuple):
                emb = emb[-1]

        if present is None:
            return emb
        output = emb.new_zeros(enroll.shape[0], *emb.shape[1:])
        output[present] = emb
        return output

    def post(self, mix_repr, emb, present=None):
        """Fuse the speaker embedding into mixture features."""
        if emb is None:
            return mix_repr
        if present is None:
            return self.fusionLayer(mix_repr, emb)

        output = mix_repr.clone()
        output[present] = self.fusionLayer(mix_repr[present], emb[present])
        return output


class SpeakerFrontend(nn.Module):
    """Instantiate the speaker features enabled by the model configuration."""

    def __init__(self, config, defer_pretrained=False):
        super().__init__()

        default_config = {
            "features": {
                "listen": {
                    "enabled": False,
                    "glue": 512,
                    "win": 512,
                    "hop": 128,
                },
                "usef": {
                    "enabled": False,
                    "causal": False,
                    "spec_dim": 2,
                    "emb_dim": 128,
                    "enc_dim": 65,
                    "approx_qk_dim": 512,
                    "n_head": 4,
                    "t_ksize": 3,
                },
                "tfmap": {
                    "enabled": False,
                    "type": "spec",
                },
                "context": {
                    "enabled": False,
                    "speaker_model": None,
                    "embed_dim": 512,
                    "atten_dim": 128,
                    "num_heads": 2,
                    "fusion": "multiply",
                    "mix_dim": 128,
                    "band": 1,
                },
                "spkemb": {
                    "enabled": False,
                    "input": "waveform",
                    "speaker_model": None,
                    "fusion": "multiply",
                    "mix_dim": 128,
                },
                "spex_plus": {
                    "enabled": False,
                    "input": "waveform",
                    "enc_dim": 256,
                    "bottleneck_dim": 256,
                    "conv_dim": 512,
                    "kernel_size": 3,
                    "num_repeats": 4,
                    "embed_dim": 256,
                    "causal": False,
                    "skip_connection": False,
                },
            },
            "speaker_model": {
                "fbank": {
                    "num_mel_bins": 80,
                    "frame_shift": 10,
                    "frame_length": 25,
                    "dither": 0.0,
                    "sample_rate": 16000,
                },
                "speaker_encoder": {
                    "model":
                    "ECAPA_TDNN_GLOB_c512",
                    "pretrained":
                    ("./wespeaker_models/voxceleb_ECAPA512/avg_model.pt"),
                    "spk_args": {
                        "embed_dim": 192,
                        "feat_dim": 80,
                        "pooling_func": "ASTP",
                    },
                },
            },
        }
        self.config = deep_update(default_config, config)
        features = self.config["features"]
        spkemb = features["spkemb"]

        # Resolve one cue input shared by every enabled speaker feature.
        enabled_inputs = {
            name: conf.get("input", "waveform")
            for name, conf in features.items() if conf["enabled"]
        }
        input_types = set(enabled_inputs.values())
        if len(input_types) > 1:
            raise ValueError(
                "Enabled speaker features require different cue inputs: "
                f"{enabled_inputs}")
        self.input_type = next(iter(input_types), "waveform")
        if self.input_type not in ("waveform", "embedding"):
            raise ValueError(
                f"Unsupported speaker cue input: {self.input_type}")

        # Share one speaker encoder unless a feature requests its own model.
        shared_context = (features["context"]["enabled"]
                          and not features["context"]["speaker_model"])
        shared_spkemb = (spkemb["enabled"] and spkemb["input"] == "waveform"
                         and not spkemb["speaker_model"])
        if shared_context or shared_spkemb:
            self.fbank = Fbank_kaldi(**self.config["speaker_model"]["fbank"])
            self.encoder = SpeakerEncoder(
                self.config["speaker_model"]["speaker_encoder"],
                defer_pretrained=defer_pretrained,
            )
        shared_fbank = getattr(self, "fbank", None)
        shared_encoder = getattr(self, "encoder", None)

        # Expose enabled features by name for the model's explicit feature path.
        if features["listen"]["enabled"]:
            self.listen = ListenFeature(features["listen"])
        if features["usef"]["enabled"]:
            self.usef = UsefFeature(features["usef"])
        if features["tfmap"]["enabled"]:
            self.tfmap = TFMapFeature(features["tfmap"])
        if features["context"]["enabled"]:
            self.context = ContextFeature(
                features["context"],
                fbank=shared_fbank,
                encoder=shared_encoder,
                defer_pretrained=defer_pretrained,
            )
        if features["spkemb"]["enabled"]:
            self.spkemb = SpeakerEmbFeature(
                features["spkemb"],
                self.config["speaker_model"],
                fbank=shared_fbank,
                encoder=shared_encoder,
                defer_pretrained=defer_pretrained,
            )
        if features["spex_plus"]["enabled"]:
            self.spex_plus = SpExPlusFeature(features["spex_plus"])
