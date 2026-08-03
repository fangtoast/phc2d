"""Representation metadata for multi-task photonic-crystal datasets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from phc2d.data.target_specs import InputRepresentation, OutputTarget


@dataclass(frozen=True)
class DataRepresentationSpec:
    """Input/output contract used by data conversion, models, and metrics."""

    input_repr: InputRepresentation
    output_target: OutputTarget
    grid_shape: tuple[int, int] | None = None
    k_points: int | None = None
    n_bands: int | None = None
    polarization: str | None = None
    field_components: tuple[str, ...] = ()
    source_repr: str | None = None
    normalization: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleMetadata:
    """Per-sample lineage shared across derived representations."""

    sample_id: str
    raw_source: str
    generator_name: str | None
    symmetry: str | None
    split: str
    grid_shape: tuple[int, int] | None
    k_path_shape: tuple[int, ...] | None
    n_bands: int | None
    field_shape: tuple[int, ...] | None
    polarization: str | None
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = ["DataRepresentationSpec", "SampleMetadata"]
