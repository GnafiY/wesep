# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFeature(nn.Module):
    """Common input parsing and time alignment for spatial features."""

    def __init__(self, input_fields, pairs):
        super().__init__()
        self.field_index = {
            name: index
            for index, name in enumerate(input_fields)
        }
        self.register_buffer("pairs", torch.as_tensor(pairs, dtype=torch.long))

    def _field(self, spatial_aux, name, required=True):
        if spatial_aux is None:
            if required:
                raise ValueError(
                    f"Spatial feature requires the {name!r} input field.")
            return None
        index = self.field_index.get(name)
        if index is None or index >= spatial_aux.shape[1]:
            if required:
                raise ValueError(
                    f"Spatial feature requires the {name!r} input field.")
            return None
        return spatial_aux[:, index]

    @staticmethod
    def _align_angle(angle, target_frames):
        """Align static or temporal angles to STFT frames periodically."""
        if angle.ndim == 1:
            return angle.unsqueeze(-1).expand(-1, target_frames)
        if angle.ndim != 2:
            raise ValueError(
                "Spatial angles must have shape [B] or [B, T], got "
                f"{tuple(angle.shape)}")
        if angle.shape[-1] == target_frames:
            return angle

        periodic = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
        periodic = F.interpolate(
            periodic,
            size=target_frames,
            mode="linear",
            align_corners=False,
        )
        return torch.atan2(periodic[:, 1], periodic[:, 0])

    def _ipd(self, mix_spec):
        first = mix_spec[:, self.pairs[:, 0]]
        second = mix_spec[:, self.pairs[:, 1]]
        phase = torch.angle(first) - torch.angle(second)
        return torch.remainder(phase + math.pi, 2 * math.pi) - math.pi


class DirectionalPhaseFeature(SpatialFeature):
    """Compute observed and target phase differences for a fixed array."""

    def __init__(
        self,
        input_fields,
        pairs,
        mic_positions,
        sample_rate,
        n_fft,
        speed_of_sound=343.0,
    ):
        super().__init__(input_fields, pairs)
        mic_positions = torch.as_tensor(mic_positions, dtype=torch.float32)
        baselines = (mic_positions[self.pairs[:, 0]] -
                     mic_positions[self.pairs[:, 1]])
        frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        self.register_buffer("baselines", baselines)
        self.register_buffer("angular_frequencies", 2 * math.pi * frequencies)
        self.speed_of_sound = speed_of_sound

    def _tpd(self, spatial_aux, target_frames):
        azimuth = self._align_angle(self._field(spatial_aux, "azimuth"),
                                    target_frames)
        elevation = self._field(spatial_aux, "elevation", required=False)
        if elevation is None:
            elevation = torch.zeros_like(azimuth)
        else:
            elevation = self._align_angle(elevation, target_frames)

        direction = torch.stack(
            [
                torch.cos(elevation) * torch.cos(azimuth),
                torch.cos(elevation) * torch.sin(azimuth),
                torch.sin(elevation),
            ],
            dim=-1,
        )
        delays = torch.einsum("btc,pc->bpt", direction, self.baselines)
        return (self.angular_frequencies.view(1, 1, -1, 1) *
                delays.unsqueeze(2) / self.speed_of_sound)


class IPDFeature(SpatialFeature):
    """Observed inter-channel phase differences."""

    def compute(self, mix_spec):
        return self._ipd(mix_spec)

    def post(self, mix_repr, feature):
        return torch.cat([mix_repr, feature], dim=1)


class CDFSpatialFeature(DirectionalPhaseFeature):
    """Cosine agreement between observed and target phase differences."""

    def compute(self, spatial_aux, mix_spec, present=None):
        feature = mix_spec.real.new_zeros(mix_spec.shape[0], len(self.pairs),
                                          *mix_spec.shape[-2:])
        if present is not None and not present.any():
            return feature
        if spatial_aux is None:
            raise ValueError("CDF requires spatial_aux.")

        valid = slice(None) if present is None else present
        ipd = self._ipd(mix_spec[valid])
        tpd = self._tpd(spatial_aux[valid], mix_spec.shape[-1])
        output = torch.cos(ipd - tpd)
        if present is None:
            return output
        feature[present] = output
        return feature

    def post(self, mix_repr, feature):
        return torch.cat([mix_repr, feature], dim=1)


class SDFSpatialFeature(DirectionalPhaseFeature):
    """Sine residual between observed and target phase differences."""

    def compute(self, spatial_aux, mix_spec, present=None):
        feature = mix_spec.real.new_zeros(mix_spec.shape[0], len(self.pairs),
                                          *mix_spec.shape[-2:])
        if present is not None and not present.any():
            return feature
        if spatial_aux is None:
            raise ValueError("SDF requires spatial_aux.")

        valid = slice(None) if present is None else present
        ipd = self._ipd(mix_spec[valid])
        tpd = self._tpd(spatial_aux[valid], mix_spec.shape[-1])
        output = torch.sin(ipd - tpd)
        if present is None:
            return output
        feature[present] = output
        return feature

    def post(self, mix_repr, feature):
        return torch.cat([mix_repr, feature], dim=1)


class DeltaSTFTFeature(SpatialFeature):
    """Real and imaginary STFT differences for microphone pairs."""

    def compute(self, mix_spec):
        delta = (mix_spec[:, self.pairs[:, 0]] - mix_spec[:, self.pairs[:, 1]])
        return torch.cat([delta.real, delta.imag], dim=1)

    def post(self, mix_repr, feature):
        return torch.cat([mix_repr, feature], dim=1)
