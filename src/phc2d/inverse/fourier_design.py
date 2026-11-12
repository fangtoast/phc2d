"""Bounded edits of full-cell Fourier binary design parameters.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np

from phc2d.data.full_cell_fourier import evaluate_real_fourier_field, periodic_cell_centers
from phc2d.data.full_cell_spec import FullCellStructureSpec


def update_fourier_binary(
    source: FullCellStructureSpec, *, sample_id: str,
    coefficients: Mapping[tuple[int, str], float], threshold: float,
    max_mode_radius: float,
) -> FullCellStructureSpec:
    """Construct an edited structure and recompute its latent normalization.

    Coefficient keys are (mode index, 'cos' or 'sin'). High-frequency modes
    are unchanged. A zero-edit returns identical raster values, while candidate
    geometry is always represented by an independently serializable spec.
    """
    if source.generator_family != "unified_fourier_binary" or source.material_kind != "binary":
        raise ValueError("expected unified Fourier binary structure")
    if not sample_id or not np.isfinite(threshold) or not np.isfinite(max_mode_radius) or max_mode_radius <= 0:
        raise ValueError("invalid sample identity, threshold or mode radius")
    payload = copy.deepcopy(source.parameters)
    modes = payload["modes"]
    changed: dict[str, float] = {"threshold": float(threshold)}
    for key, value in coefficients.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("coefficient key must be (index, cos/sin)")
        index, component = key
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(modes):
            raise ValueError("invalid Fourier mode index")
        mode = modes[index]
        if component not in ("cos", "sin") or not np.isfinite(value):
            raise ValueError("invalid Fourier coefficient")
        if np.hypot(int(mode["mx"]), int(mode["my"])) > max_mode_radius:
            raise ValueError("mode lies outside the configured radius")
        mode[component] = float(value)
        changed[f"modes[{index}].{component}"] = float(value)
    resolution = int(payload["normalization_grid_resolution"])
    x, y = periodic_cell_centers((resolution, resolution))
    field = evaluate_real_fourier_field(x, y, modes)
    mean, std = float(np.mean(field, dtype=np.float64)), float(np.std(field, dtype=np.float64))
    if not np.isfinite(std) or std <= 1e-12:
        raise ValueError("edited Fourier field has zero or invalid variance")
    payload["normalization_mean"], payload["normalization_std"] = mean, std
    payload["threshold"] = float(threshold)
    payload["coefficient_derivation"] = "inverse_edited_from_source_v1"
    changed.update(normalization_mean=mean, normalization_std=std)
    # This is the realized fraction on the recorded normalization grid, not a
    # request to reselect the threshold by quantile.
    fraction = float(np.mean((field - mean) / std >= threshold))
    payload["target_material_fraction"] = fraction
    changed["target_material_fraction"] = fraction
    values = list(source.parameter_vector)
    for name, value in changed.items():
        if name not in source.parameter_names:
            raise ValueError(f"missing parameter identity: {name}")
        values[source.parameter_names.index(name)] = value
    metadata = {"inverse_design": {"source_parameter_hash": source.parameter_hash,
                                   "max_mode_radius": float(max_mode_radius)}}
    record = source.to_mapping()
    record.update(sample_id=sample_id, parameters=payload, parameter_vector=values, metadata=metadata)
    return FullCellStructureSpec.from_mapping(record)
