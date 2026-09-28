# Copyright (c) 2026 Haoyu Li (haoyu.li.cs@sjtu.edu.cn)
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch
import torch.nn as nn

from wesep.models import get_model


class _FakeKCE(nn.Module):
    """Small differentiable KCE replacement for separator contract tests."""

    def __init__(self, config, defer_pretrained=False):
        super().__init__()
        del defer_pretrained
        self.output_dim = config.get("adapter", {}).get("output_dim", 8)
        self.projection = nn.Linear(2, self.output_dim)

    def forward(self, mixture, textual_aux):
        del textual_aux
        statistics = torch.stack(
            [mixture.mean(dim=-1), mixture.square().mean(dim=-1)], dim=-1)
        return self.projection(statistics)


def _assert_textual_model_forward_and_backward(model_name, config):
    with patch("wesep.modules.textual.textual_frontend.KCE", _FakeKCE):
        model = get_model(model_name)(config, defer_pretrained=True)
    batch = {
        "wav_mix": torch.randn(1, 1, 2048),
        "textual_aux": torch.tensor([[3, 4, 5, 6]]),
    }
    speech = model(batch)["speech"]
    assert speech.shape == (1, 1, 2048)
    speech.square().mean().backward()
    assert any(parameter.grad is not None
               for parameter in model.sep_model.parameters()
               if parameter.requires_grad)
    assert any(parameter.grad is not None
               for parameter in model.textual_ft.parameters()
               if parameter.requires_grad)


def test_dpccn_textual_forward_and_backward():
    _assert_textual_model_forward_and_backward(
        "TSE_DPCCN_TEXTUAL",
        {
            "separator": {
                "win": 512,
                "stride": 128,
                "nspk": 1,
                "tcn_layers": 1,
                "tcn_blocks": 1,
                "pool_sizes": (2, 4, 8, 16),
            },
            "textual": {
                "features": {
                    "dae_kce": {
                        "adapter": {
                            "output_dim": 8,
                        },
                    },
                },
            },
        },
    )


def test_tfgridnet_textual_forward_and_backward():
    _assert_textual_model_forward_and_backward(
        "TSE_TFGRIDNET_TEXTUAL",
        {
            "separator": {
                "nspk": 1,
                "n_fft": 128,
                "stride": 64,
                "n_layers": 1,
                "hidden_channels": 32,
                "n_head": 2,
                "approx_qk_dim": 64,
                "emb_dim": 16,
                "emb_ks": 4,
                "emb_hs": 1,
            },
            "textual": {
                "features": {
                    "dae_kce": {
                        "adapter": {
                            "output_dim": 8,
                        },
                    },
                },
            },
        },
    )
