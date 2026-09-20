# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0

import itertools

import torch.nn as nn

from wesep.modules.common.deep_update import deep_update
from wesep.modules.spatial.doa_features import (
    CyclicDOAFeature,
    InitStateDOAFeature,
)
from wesep.modules.spatial.fixed_features import (
    CDFSpatialFeature,
    DeltaSTFTFeature,
    IPDFeature,
    SDFSpatialFeature,
)


class SpatialFrontend(nn.Module):
    """Build multichannel observation and target-direction features."""

    def __init__(
        self,
        config,
        sample_rate=16000,
        n_fft=512,
        mix_dim=128,
        num_layers=1,
        num_mics=None,
    ):
        super().__init__()

        default_config = {
            "input_fields": ["azimuth", "elevation"],
            "array": {
                "mic_positions": None
            },
            "pairs": "all",
            "features": {
                "ipd": {
                    "enabled": False
                },
                "cdf": {
                    "enabled": False
                },
                "sdf": {
                    "enabled": False
                },
                "delta_stft": {
                    "enabled": False
                },
                "cyc_doaemb": {
                    "enabled": False,
                    "alpha": 20,
                    "encoding_dim": 40,
                    "use_elevation": True,
                    "feature_dim": mix_dim,
                    "fusion": "multiply",
                },
                "initstate_emb": {
                    "enabled": False,
                    "alpha": 20,
                    "encoding_dim": 40,
                    "use_elevation": True,
                },
            },
        }
        self.config = deep_update(default_config, config)
        input_fields = self.config["input_fields"]
        mic_positions = self.config["array"]["mic_positions"]
        if num_mics is None:
            if mic_positions is None:
                raise ValueError(
                    "Spatial frontend requires num_mics or mic_positions.")
            num_mics = len(mic_positions)
        if mic_positions is not None and len(mic_positions) != num_mics:
            raise ValueError(
                f"Configured array has {len(mic_positions)} microphones, "
                f"but the mixture has {num_mics} channels.")

        features = self.config["features"]
        pair_features = ("ipd", "cdf", "sdf", "delta_stft")
        uses_pairs = any(features[name]["enabled"] for name in pair_features)
        uses_direction_geometry = any(features[name]["enabled"]
                                      for name in ("cdf", "sdf"))
        if uses_direction_geometry and mic_positions is None:
            raise ValueError(
                "CDF and SDF require array.mic_positions in the model config.")

        # Resolve microphone pairs independently from target-direction cues.
        pairs = self.config["pairs"]
        if pairs == "all":
            pairs = list(itertools.combinations(range(num_mics), 2))
        if uses_pairs and not pairs:
            raise ValueError(
                "Spatial frontend requires at least one mic pair.")
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError(
                f"Microphone pairs must contain two indices: {pairs}")
        if any(index < 0 or index >= num_mics for pair in pairs
               for index in pair):
            raise ValueError(
                f"Microphone pairs exceed the {num_mics}-channel array: "
                f"{pairs}")
        self.pairs = pairs
        self.num_mics = num_mics
        self.spectral_channels = 0

        feature_args = {
            "input_fields": input_fields,
            "pairs": pairs,
        }
        direction_args = {
            **feature_args,
            "mic_positions": mic_positions,
            "sample_rate": sample_rate,
            "n_fft": n_fft,
        }
        # Expose enabled features by name for explicit model insertion paths.
        if features["ipd"]["enabled"]:
            self.ipd = IPDFeature(**feature_args)
            self.spectral_channels += len(pairs)
        if features["cdf"]["enabled"]:
            self.cdf = CDFSpatialFeature(**direction_args)
            self.spectral_channels += len(pairs)
        if features["sdf"]["enabled"]:
            self.sdf = SDFSpatialFeature(**direction_args)
            self.spectral_channels += len(pairs)
        if features["delta_stft"]["enabled"]:
            self.delta_stft = DeltaSTFTFeature(**feature_args)
            self.spectral_channels += 2 * len(pairs)
        if features["cyc_doaemb"]["enabled"]:
            self.cyc_doaemb = CyclicDOAFeature(
                features["cyc_doaemb"],
                input_fields,
                pairs,
                mix_dim,
            )
        if features["initstate_emb"]["enabled"]:
            self.initstate_emb = InitStateDOAFeature(
                features["initstate_emb"],
                input_fields,
                pairs,
                mix_dim,
                num_layers,
            )
