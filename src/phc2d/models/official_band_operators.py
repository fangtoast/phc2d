"""Pinned NeuralOperator backbones with a shared Full-Wh output adapter.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

from importlib.metadata import version
from typing import Any

from phc2d.models.torch_support import image_contract, make_band_head, require_torch, target_contract

OFFICIAL_VERSION = "2.0.0"
OFFICIAL_COMMIT = "b307030d9b7ee530823d6a05bddb8238ba22a743"


def build_official_band_operator(
    *, input_shape: Any, output_shape: Any, tensorized: bool,
    width: int = 64, depth: int = 4, modes: Any = (12, 12), rank: float = .1,
) -> Any:
    """Adapt the official field model, preserving its unmodified computational modules."""
    torch, nn = require_torch("official_band_operator")
    try:
        if version("neuraloperator") != OFFICIAL_VERSION:
            raise ImportError(f"Expected neuraloperator=={OFFICIAL_VERSION}")
        from neuralop.models import FNO, TFNO
    except ImportError as exc:
        raise ImportError("Install phc2d[official] with neuraloperator==2.0.0 and its dependencies") from exc
    _, channels = image_contract(input_shape, model_name="official_band_operator")
    target, _, _ = target_contract(output_shape)
    ky, kx = (int(v) for v in modes)
    if min(ky, kx) <= 0 or kx < 2 or int(width) <= 0 or int(depth) <= 0:
        raise ValueError("positive model dimensions and modes_x>=2 are required")
    factory = TFNO if tensorized else FNO

    class OfficialBandOperator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape, self.output_shape = tuple(input_shape), target
            self.official_version, self.official_commit = OFFICIAL_VERSION, OFFICIAL_COMMIT
            self.backbone = factory(
                n_modes=(2 * ky, 2 * (kx - 1)), in_channels=channels, out_channels=width,
                hidden_channels=width, n_layers=depth, positional_embedding=None,
                lifting_channel_ratio=0, use_channel_mlp=True, channel_mlp_expansion=.5,
                norm=None, domain_padding=None, implementation="factorized",
                factorization="Tucker" if tensorized else None, rank=rank if tensorized else 1.,
            )
            self.backbone.projection = nn.Identity()
            self.head = make_band_head(
                torch=torch, nn=nn, latent_dim=width, output_shape=target, head_kind="dense",
                k_points=None, head_hidden_dim=256, band_embedding_dim=32,
                fourier_frequencies=(1, 2, 4, 8), dropout=0.,
            )

        def forward(self, x: Any) -> Any:
            if x.ndim == 3:
                x = x.unsqueeze(1)
            return self.head(self.backbone(x).mean(dim=(-2, -1)))

        def state_dict(self, *args: Any, **kwargs: Any) -> Any:
            # NeuralOperator adds a non-tensor '_metadata' entry to nested exports.
            # Keep checkpoints compatible with strict PyTorch state-dict loading.
            state = super().state_dict(*args, **kwargs)
            state.pop("_metadata", None)
            return state

    return OfficialBandOperator()


def build_official_fno_band_v1(**kwargs: Any) -> Any:
    return build_official_band_operator(tensorized=False, **kwargs)


def build_official_tucker_tfno_band_v1(**kwargs: Any) -> Any:
    return build_official_band_operator(tensorized=True, **kwargs)
