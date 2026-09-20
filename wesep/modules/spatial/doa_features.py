# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0

import warnings

import torch
import torch.nn as nn

from wesep.modules.fusion.speech import SpeakerFuseLayer
from wesep.modules.spatial.fixed_features import SpatialFeature
from wesep.modules.spatial.pos_encoding import CycPosEncoding


class CyclicDOAFeature(SpatialFeature):
    """Encode static or moving source directions with a trainable MLP."""

    def __init__(self, config, input_fields, pairs, mix_dim):
        super().__init__(input_fields, pairs)
        self.use_elevation = config.get("use_elevation", True)
        encoding_dim = config.get("encoding_dim", 40)
        feature_dim = config.get("feature_dim", mix_dim)

        self.position_encoding = CycPosEncoding(
            embed_dim=encoding_dim,
            alpha=config.get("alpha", 20),
        )
        input_dim = encoding_dim * (2 if self.use_elevation else 1)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.PReLU(),
        )
        self.fusion = SpeakerFuseLayer(
            embed_dim=feature_dim,
            feat_dim=mix_dim,
            fuse_type=config.get("fusion", "multiply"),
        )

    def compute(self, spatial_aux, mix_repr, present=None):
        feature = mix_repr.new_zeros(mix_repr.shape[0], 1,
                                     self.projection[0].out_features,
                                     mix_repr.shape[-1])
        if present is not None and not present.any():
            return feature
        if spatial_aux is None:
            raise ValueError("Cyclic DOA feature requires spatial_aux.")

        valid = slice(None) if present is None else present
        target_frames = mix_repr.shape[-1]
        azimuth = self._align_angle(self._field(spatial_aux[valid], "azimuth"),
                                    target_frames)
        encoded = [self.position_encoding(azimuth)]

        if self.use_elevation:
            elevation = self._field(
                spatial_aux[valid],
                "elevation",
                required=False,
            )
            if elevation is None:
                elevation = torch.zeros_like(azimuth)
            else:
                elevation = self._align_angle(elevation, target_frames)
            encoded.append(self.position_encoding(elevation))

        valid_feature = self.projection(torch.cat(encoded, dim=-1))
        valid_feature = valid_feature.transpose(1, 2).unsqueeze(1)
        if present is None:
            return valid_feature
        feature[present] = valid_feature
        return feature

    def post(self, mix_repr, feature, present=None):
        if present is None:
            return self.fusion(mix_repr, feature)
        if not present.any():
            return mix_repr

        output = mix_repr.clone()
        output[present] = self.fusion(mix_repr[present], feature[present])
        return output


class InitStateDOAFeature(SpatialFeature):
    """Map the initial target direction to BSRNN recurrent states."""

    def __init__(self, config, input_fields, pairs, mix_dim, num_layers):
        super().__init__(input_fields, pairs)
        self.use_elevation = config.get("use_elevation", True)
        encoding_dim = config.get("encoding_dim", 40)
        self.position_encoding = CycPosEncoding(
            embed_dim=encoding_dim,
            alpha=config.get("alpha", 20),
        )
        input_dim = encoding_dim * (2 if self.use_elevation else 1)
        hidden_dim = mix_dim * 2
        self.projections = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim * 4) for _ in range(num_layers)])
        self._warned_temporal_input = False

    def compute(self, spatial_aux, mix_repr=None, present=None):
        if present is not None and not present.any():
            state_dim = self.projections[0].out_features // 4
            zero = self.projections[0].weight.new_zeros(
                present.shape[0], state_dim)
            return [{
                "band": (zero, zero),
                "communication": (zero, zero),
            } for _ in self.projections]
        if spatial_aux is None:
            raise ValueError("Init-state DOA feature requires spatial_aux.")

        valid = slice(None) if present is None else present
        valid_spatial = spatial_aux[valid]
        azimuth = self._field(valid_spatial, "azimuth")
        if azimuth.ndim == 2:
            if (azimuth.shape[-1] > 1 and not self._warned_temporal_input):
                warnings.warn(
                    "initstate_emb uses only the first frame of temporal DOA "
                    "cues.",
                    stacklevel=2,
                )
                self._warned_temporal_input = True
            azimuth = azimuth[:, 0]

        encoded = [self.position_encoding(azimuth)]

        if self.use_elevation:
            elevation = self._field(valid_spatial, "elevation", required=False)
            if elevation is None:
                elevation = torch.zeros_like(azimuth)
            elif elevation.ndim == 2:
                elevation = elevation[:, 0]
            encoded.append(self.position_encoding(elevation))

        direction = torch.cat(encoded, dim=-1)
        states = []
        for projection in self.projections:
            band_h, band_c, comm_h, comm_c = projection(direction).chunk(4, -1)
            if present is not None:
                valid_states = (band_h, band_c, comm_h, comm_c)
                full_states = [
                    state.new_zeros(spatial_aux.shape[0], state.shape[-1])
                    for state in valid_states
                ]
                for full_state, valid_state in zip(full_states, valid_states):
                    full_state[present] = valid_state
                band_h, band_c, comm_h, comm_c = full_states
            states.append({
                "band": (band_h, band_c),
                "communication": (comm_h, comm_c),
            })
        return states

    def post(self, mix_repr, feature):
        return mix_repr
