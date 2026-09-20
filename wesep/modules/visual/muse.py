# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
"""MuSE-inspired visual feature extraction for WeSep.

Reference:
    Z. Pan, R. Tao, C. Xu, and H. Li, "MuSE: Multi-Modal Target
    Speaker Extraction with Visual Cues," ICASSP 2021.

Reference implementation:
    https://github.com/zexupan/MuSE

The lip preprocessing, visual encoder, and temporal visual convolution follow
the MuSE architecture. The configurable input stage, deferred checkpoint
loading, and per-BSNet fusion are WeSep adaptations.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.fusion.speech import SpeakerFuseLayer
from wesep.utils.torch_compat import register_load_state_dict_pre_hook


class Muse_LipROIProcessor(nn.Module):
    """Convert RGB video ``[B,H,W,3,T]`` into normalized lip ROIs."""

    def __init__(self,
                 roi_size=112,
                 resize_size=224,
                 norm_mean=0.4161,
                 norm_std=0.1688):
        super().__init__()
        self.roi_size = roi_size
        self.resize_size = resize_size
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def forward(self, video):
        B, H, W, C, T = video.shape
        if C != 3:
            raise ValueError(f"Expected three RGB channels, got {C}")
        if video.dtype not in (torch.uint8, torch.float32):
            raise TypeError(
                f"raw video must be uint8 or float32, got {video.dtype}")

        # Convert compact RGB frames into normalized grayscale images.
        video = video.permute(0, 4, 3, 1, 2).contiguous()
        scale = 1.0 / 255.0 if video.dtype == torch.uint8 else 1.0
        gray = video[:, :, 0].float().mul_(0.299 * scale)
        gray.add_(video[:, :, 1], alpha=0.587 * scale)
        gray.add_(video[:, :, 2], alpha=0.114 * scale)

        # Resize each frame and take the centered lip region.
        gray = gray.view(B * T, 1, H, W)
        if H != self.resize_size or W != self.resize_size:
            gray = F.interpolate(
                gray,
                size=(self.resize_size, self.resize_size),
                mode="bilinear",
                align_corners=False,
            )
        start = self.resize_size // 2 - self.roi_size // 2
        roi = gray[:, :, start:start + self.roi_size,
                   start:start + self.roi_size]
        roi = roi.view(B, T, self.roi_size, self.roi_size)
        return (roi - self.norm_mean) / self.norm_std


class ResNetLayer(nn.Module):
    """Two residual units used by the MuSE visual ResNet."""

    def __init__(self, inplanes, outplanes, stride):
        super().__init__()
        self.conv1a = nn.Conv2d(inplanes,
                                outplanes,
                                3,
                                stride=stride,
                                padding=1,
                                bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, 3, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes,
                                    outplanes,
                                    1,
                                    stride=stride,
                                    bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, 3, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, 3, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

    def forward(self, x):
        residual = x if self.stride == 1 else self.downsample(x)
        x = self.conv2a(F.relu(self.bn1a(self.conv1a(x))))
        x = x + residual
        intermediate = x
        x = F.relu(self.outbna(x))

        x = self.conv2b(F.relu(self.bn1b(self.conv1b(x))))
        return F.relu(self.outbnb(x + intermediate))


class ResNet(nn.Module):
    """ResNet-18-style frame encoder used by the visual frontend."""

    def __init__(self):
        super().__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4, 4), stride=(1, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x)


class Muse_VisualFrontend(nn.Module):
    """Encode each lip frame into a 512-dimensional representation."""

    def __init__(self):
        super().__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                64,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )
        self.resnet = ResNet()

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        batch_size = x.shape[0]
        x = self.frontend3D(x).transpose(1, 2)
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.resnet(x)
        return x.reshape(batch_size, -1, 512).transpose(1, 2)


def load_muse_frontend(frontend, checkpoint):
    """Load MuSE frame-encoder parameters into a visual frontend."""
    state_dict = torch.load(checkpoint, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    frontend.load_state_dict(state_dict, strict=True)


def compute_muse_frontend(video, roi, frontend):
    """Convert raw video ``[B,H,W,3,T]`` into ``[B,512,T]`` features."""
    lip_roi = roi(video)
    return frontend(lip_roi.transpose(0, 1).unsqueeze(2))


class Muse_VisualConv1D(nn.Module):
    """One residual temporal visual-convolution block."""

    def __init__(self, channels=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.PReLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x) + x


class DirectVisualAdapter(nn.Module):
    """Pass frame-level visual features through unchanged."""

    def __init__(self, config):
        super().__init__()
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]

    def forward(self, x):
        return x


class MuseVisualAdapter(nn.Module):
    """Apply the temporal visual-convolution stack used by MuSE."""

    def __init__(self, config):
        super().__init__()
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.net = nn.Sequential(*[
            Muse_VisualConv1D(channels=self.output_dim)
            for _ in range(config["num_layers"])
        ])

    def forward(self, x):
        return self.net(x)


class ClearerVisualConv1D(nn.Module):
    """One ClearerVoice-style residual temporal convolution block."""

    def __init__(self, channels=256, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, hidden_dim, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                3,
                padding=1,
                groups=hidden_dim,
                bias=True,
            ),
            nn.PReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, channels, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x) + x


class ClearerVisualAdapter(nn.Module):
    """Project and process visual cues with ClearerVoice-style VTCN blocks."""

    def __init__(self, config):
        super().__init__()
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.net = nn.Sequential(
            nn.Conv1d(
                self.input_dim,
                self.output_dim,
                1,
                bias=False,
            ),
            *[
                ClearerVisualConv1D(
                    channels=self.output_dim,
                    hidden_dim=config["hidden_dim"],
                ) for _ in range(config["num_layers"])
            ],
        )

    def forward(self, x):
        return self.net(x)


def build_visual_adapter(config):
    """Build the configured temporal visual adapter."""
    adapter_type = config["type"]
    if adapter_type == "direct":
        return DirectVisualAdapter(config)
    if adapter_type == "muse":
        return MuseVisualAdapter(config)
    if adapter_type == "clearer":
        return ClearerVisualAdapter(config)
    raise ValueError(f"Unsupported visual adapter: {adapter_type}")


class MuseVisualFeature(nn.Module):
    """Build and fuse MuSE-like features from raw or precomputed cues."""

    def __init__(self, config, defer_pretrained=False):
        super().__init__()
        self.input_type = config["input"]
        self.pretrained = config["vf_pretrained"]
        self.pretrained_loaded = False
        self.fallback_reported = False
        self.freeze_frontend = config.get("freeze_frontend", True)
        self.multi_fuse = config["multi_fuse"]

        # Raw video includes the pretrained frame-level visual encoder.
        if self.input_type == "raw_video":
            self.roi = Muse_LipROIProcessor()
            self.visual_frontend = Muse_VisualFrontend()
            if self.pretrained is not None and not defer_pretrained:
                self.load_pretrained()
            register_load_state_dict_pre_hook(self.visual_frontend,
                                              self._prepare_pretrained_state)
            if self.freeze_frontend:
                self.visual_frontend.requires_grad_(False)
                self.visual_frontend.eval()

        self.adapter = build_visual_adapter(config["adapter"])
        self.upsample_to_audio = config.get("upsample", False)

        # Multi-fuse assigns an independent projection to every repeat.
        self.fusion_layers = nn.ModuleList([
            SpeakerFuseLayer(
                embed_dim=self.adapter.output_dim,
                feat_dim=config["mix_dim"],
                fuse_type=config["fusion"],
            ) for _ in range(config["repeat"] if self.multi_fuse else 1)
        ])

    def load_pretrained(self):
        """Load the external MuSE encoder checkpoint at most once."""
        if self.pretrained_loaded:
            return
        if self.pretrained is None:
            raise RuntimeError(
                "The TSE checkpoint has no MuSE visual encoder parameters "
                "and vf_pretrained is not configured.")
        load_muse_frontend(self.visual_frontend, self.pretrained)
        self.pretrained_loaded = True

    def _prepare_pretrained_state(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Use external weights when the TSE checkpoint omits this encoder."""
        local_state = self.visual_frontend.state_dict()
        if any(prefix + key in state_dict for key in local_state):
            return

        self.load_pretrained()
        for key, value in self.visual_frontend.state_dict().items():
            state_dict[prefix + key] = value

        if not self.fallback_reported:
            logging.warning(
                "MuSE visual encoder is absent from the TSE checkpoint; "
                "loaded external pretrained parameters from %s.",
                self.pretrained,
            )
            self.fallback_reported = True

    def compute(self, video, mix=None):
        """Return MuSE features ``[B,D,T_v]`` from the configured input."""
        if self.input_type == "raw_video":
            with torch.set_grad_enabled(not self.freeze_frontend):
                feat = compute_muse_frontend(video, self.roi,
                                             self.visual_frontend)
        else:
            feat = video.flatten(1, -2)
            if feat.shape[1] != self.adapter.input_dim:
                raise ValueError("Precomputed MuSE features must provide "
                                 f"{self.adapter.input_dim} channels, "
                                 f"got {tuple(video.shape)}")

        feat = self.adapter(feat)
        if self.upsample_to_audio and mix is not None:
            feat = F.interpolate(feat, size=mix.shape[-1], mode="linear")
        return feat

    def post(self, mix_repr, feat_repr, stage):
        """Inject the visual representation before one separator block."""
        return self.fusion_layers[stage](mix_repr, feat_repr)

    def train(self, mode=True):
        """Keep a frozen MuSE encoder in evaluation mode during training."""
        super().train(mode)
        if self.input_type == "raw_video" and self.freeze_frontend:
            self.visual_frontend.eval()
        return self
