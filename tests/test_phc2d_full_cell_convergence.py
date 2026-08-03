"""Tests for full-cell resolution selection and metrics.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from phc2d.data.full_cell import FullCellStructureSpec, build_full_bz_grid
from phc2d.data.full_cell_convergence import (
    compute_convergence_summary,
    representative_k_indices,
    select_convergence_indices,
)


def _structure(index: int, stratum: str) -> FullCellStructureSpec:
    if stratum.startswith("graded"):
        family = "graded_rbf_fourier"
        material_kind = "continuous"
        metadata = {"core_count": int(stratum[-1])}
        parameters = {
            "blobs": [], "modes": [], "fourier_strength": 0.0,
            "temperature": 1.0, "bias": float(index), "target_material_fraction": 0.5,
        }
    elif stratum.startswith("polar"):
        family = "binary_polar_holes"
        material_kind = "binary"
        metadata = {"hole_count": int(stratum[-1])}
        parameters = {"holes": [], "hole_count": int(stratum[-1])}
    else:
        family = "binary_spectral_threshold"
        material_kind = "binary"
        metadata = {}
        parameters = {"modes": [], "threshold": 0.0}
    return FullCellStructureSpec(
        sample_id=f"sample_{index:03d}",
        generator_family=family,
        material_kind=material_kind,
        parameters=parameters,
        parameter_names=("value",),
        parameter_vector=(float(index),),
        parameter_mask=(True,),
        metadata=metadata,
    )


def test_selection_round_robins_generator_strata_deterministically() -> None:
    structures = [
        _structure(0, "graded1"),
        _structure(1, "graded1"),
        _structure(2, "graded2"),
        _structure(3, "graded2"),
    ]
    first = select_convergence_indices(structures, 4)
    second = select_convergence_indices(structures, 4)

    assert first.tolist() == second.tolist()
    assert {structures[int(index)].metadata["core_count"] for index in first[:2]} == {1, 2}


def test_representative_k_indices_select_nine_full_bz_locations() -> None:
    _, grid_index = build_full_bz_grid(9)
    selected = representative_k_indices(grid_index)

    assert selected.shape == (9,)
    assert {tuple(row) for row in grid_index[selected]} == {
        (-4, -4), (-4, 0), (-4, 4),
        (0, -4), (0, 0), (0, 4),
        (4, -4), (4, 0), (4, 4),
    }


def test_convergence_summary_excludes_first_band_and_applies_gates() -> None:
    base = np.ones((2, 9, 4), dtype=np.float64)
    arrays = {
        64: base.copy(),
        96: base.copy(),
        128: base.copy(),
    }
    arrays[64][..., 0] = 1000.0
    arrays[96][..., 1:] *= 1.002
    arrays[128][..., 1:] *= 1.003
    summary = compute_convergence_summary(
        arrays,
        median_64_96_max=0.005,
        median_96_128_max=0.0025,
        p95_max=0.01,
    )

    assert summary["passed"] is True
    assert summary["bands_in_gate"] == [2, 3, 4]
    assert summary["median_mre_64_96"] < 0.005
    assert summary["median_mre_96_128"] < 0.0025
