"""Channel-wise learned fusion for a separately identified Global-Local model.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

from typing import Any

from phc2d.models.full_cell_operators import build_factorized_band_operator


def make_channel_fusion(nn: Any, torch: Any, channels: int) -> Any:
    """Create two normalized channel gates without consuming initialization RNG."""
    class ChannelFusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.channel_logits = nn.Parameter(torch.zeros(2, channels))

        def normalized_weights(self) -> Any:
            return self.channel_logits.softmax(dim=0)

        def forward(self, branches: list[Any], names: list[str]) -> Any:
            indices = [("spectral", "local").index(name) for name in names]
            weights = self.channel_logits[indices].softmax(dim=0)
            return sum(w[None, :, None, None] * branch for w, branch in zip(weights, branches))

    return ChannelFusion()


def build_global_local_channel_gate_v1(
    *, input_shape: Any, output_shape: Any, width: int = 64, depth: int = 6,
    modes: Any = (12, 12), factorization_rank: float = .75, norm_groups: int = 8,
    channel_expansion: int = 4, local_kernel: int = 3, drop_path: float = .1,
    dropout: float = .05,
) -> Any:
    """Preserve the GL backbone and replace only fixed branch averaging."""
    model = build_factorized_band_operator(
        model_name="global_local_channel_gate_v1", input_shape=input_shape,
        output_shape=output_shape, width=width, depth=depth, modes=modes,
        factorization_rank=factorization_rank, norm_groups=norm_groups,
        channel_expansion=channel_expansion, local_kernel=local_kernel,
        drop_path=drop_path, head_kind="dense", k_points=None,
        head_hidden_dim=256, band_embedding_dim=32, fourier_frequencies=(1, 2, 4, 8),
        dropout=dropout, lane_conditioned=False, num_lanes=2, lane_embedding_dim=16,
        enabled_branches=("spectral", "local"), local_operator_kind="convnext",
        branch_fusion="fixed_mean", fusion_factory=make_channel_fusion,
    )
    model.__class__.__phc2d_architecture_module__ = __name__
    return model
