"""Task target names and lightweight target metadata.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

InputRepresentation = Literal[
    "rho_wedge",
    "rho_grid",
    "epsilon_grid",
    "image_binary",
    "image_epsilon",
    "param_vector",
    "multimodal",
    "field_conditioned",
]

OutputTarget = Literal[
    "Wh",
    "band_query",
    "gap_descriptors",
    "near_field_te",
    "near_field_tm",
    "near_field_te_tm",
    "scalar_fom",
    "multi_task",
]


@dataclass(frozen=True)
class TargetSpec:
    """Metadata describing one supervised or evaluation target."""

    output_target: OutputTarget
    shape: tuple[int, ...] | None = None
    k_points: int | None = None
    n_bands: int | None = None
    polarization: str | None = None
    field_components: tuple[str, ...] = ()
    units: str | None = None
    normalization: dict[str, Any] = field(default_factory=dict)


__all__ = ["InputRepresentation", "OutputTarget", "TargetSpec"]
