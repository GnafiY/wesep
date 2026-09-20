# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Reference:
#   K. Zhang et al., "Multi-Level Speaker Representation for Target Speaker
#   Extraction," arXiv:2410.16059, 2024.

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.fusion.speech import CrossFuse, SpeakerFuseLayer
from wesep.modules.speaker.encoder import Fbank_kaldi, SpeakerEncoder


class TFMapFeature(nn.Module):
    """Build and concatenate the speaker TF-map representation."""

    def __init__(self, config):
        super().__init__()

    def compute(self, enroll, mix, present=None):
        """Compute a TF-map from enrollment and mixture magnitudes ``(B,F,T)``."""
        output = mix.new_zeros(mix.shape)
        if present is not None and not present.any():
            return output

        valid = slice(None) if present is None else present
        mix_valid = mix[valid]
        enroll_valid = enroll[valid]
        mix_mag = F.normalize(mix_valid, p=2, dim=1)
        enroll_mag = F.normalize(enroll_valid, p=2, dim=1)

        mix_mag = mix_mag.permute(0, 2, 1).contiguous()
        att_scores = torch.matmul(mix_mag, enroll_mag)
        att_weights = F.softmax(att_scores, dim=-1)
        enroll_mag = enroll_mag.permute(0, 2, 1).contiguous()
        tf_map = torch.matmul(att_weights, enroll_mag)
        tf_map = tf_map.permute(0, 2, 1).contiguous()

        tf_map = F.normalize(tf_map, p=2, dim=1)
        feature = (torch.sum(mix_valid * tf_map, dim=1, keepdim=True) * tf_map)
        if present is None:
            return feature
        output[present] = feature
        return output

    def post(self, mix_repr, feat_repr):
        """Concatenate ``(B,1,F,T)`` TF-map features with mixture features."""
        return torch.cat([mix_repr, feat_repr], dim=1)


class ContextFeature(nn.Module):
    """Extract and fuse frame-level speaker context features."""

    def __init__(
        self,
        conf_context,
        fbank=None,
        encoder=None,
        defer_pretrained=False,
    ):
        super().__init__()
        if conf_context["speaker_model"]:
            self.fbank = Fbank_kaldi(**conf_context["speaker_model"]["fbank"])
            self.encoder = SpeakerEncoder(
                conf_context["speaker_model"]["speaker_encoder"],
                defer_pretrained=defer_pretrained,
            )
        else:
            self.fbank = fbank
            self.encoder = encoder

        self.attenFuse = CrossFuse(
            embed_dim=conf_context["embed_dim"],
            atten_dim=conf_context["atten_dim"],
            mix_dim=conf_context["mix_dim"],
            num_heads=conf_context["num_heads"],
            nband=conf_context["band"],
            batch_first=True,
        )
        self.fusionLayer = SpeakerFuseLayer(
            embed_dim=conf_context["atten_dim"],
            feat_dim=conf_context["mix_dim"],
            fuse_type=conf_context["fusion"],
        )

    def compute(self, enroll, mix=None, present=None):
        """Extract frame-level speaker features ``(B,F_e,T_e)``."""
        if present is not None and not present.any():
            return None

        valid_enroll = enroll if present is None else enroll[present]
        fb = self.fbank(valid_enroll)
        emb = self.encoder.spk_model._get_frame_level_feat(fb)
        if isinstance(emb, tuple):
            emb = emb[-1]
        if present is None:
            return emb

        output = emb.new_zeros(enroll.shape[0], *emb.shape[1:])
        output[present] = emb
        return output

    def post(self, mix_repr, emb, present=None):
        """Fuse frame-level speaker context into mixture features."""
        if emb is None:
            return mix_repr
        if present is None:
            emb = self.attenFuse(mix_repr, emb)
            return self.fusionLayer(mix_repr, emb)

        output = mix_repr.clone()
        valid_mix = mix_repr[present]
        valid_emb = self.attenFuse(valid_mix, emb[present])
        output[present] = self.fusionLayer(valid_mix, valid_emb)
        return output
