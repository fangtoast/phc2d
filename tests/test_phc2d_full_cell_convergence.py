"""Tests for full-cell resolution selection and metrics.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.data.full_cell import FullCellStructureSpec, build_full_bz_grid
from phc2d.data.full_cell_convergence import (
    await_full_cell_pilots_and_finalize,
    compute_convergence_summary,
    compute_reference_convergence_summary,
    representative_k_indices,
    select_convergence_indices,
    select_convergence_indices_by_stratum,
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


def test_selection_honors_exact_stratum_counts() -> None:
    structures = [
        *[_structure(index, "polar1") for index in range(20)],
        *[_structure(index + 20, "polar2") for index in range(20)],
        *[_structure(index + 40, "spectral") for index in range(40)],
    ]

    selected = select_convergence_indices_by_stratum(
        structures,
        {
            "binary_polar_holes_1": 13,
            "binary_polar_holes_2": 12,
            "binary_spectral_threshold": 25,
        },
    )
    selected_structures = [structures[int(index)] for index in selected]

    assert selected.shape == (50,)
    assert sum(item.metadata.get("hole_count") == 1 for item in selected_structures) == 13
    assert sum(item.metadata.get("hole_count") == 2 for item in selected_structures) == 12
    assert sum(item.generator_family == "binary_spectral_threshold" for item in selected_structures) == 25


def test_selection_uses_serialized_parameters_when_metadata_omits_counts() -> None:
    polar = [replace(_structure(index, f"polar{1 + index % 2}"), metadata={}) for index in range(8)]
    graded = [
        replace(
            _structure(index + 20, f"graded{1 + index % 2}"),
            metadata={},
            parameters={
                **_structure(index + 20, f"graded{1 + index % 2}").parameters,
                "blobs": [{}] * (1 + index % 2),
            },
        )
        for index in range(8)
    ]

    assert select_convergence_indices_by_stratum(
        polar,
        {"binary_polar_holes_1": 2, "binary_polar_holes_2": 2},
    ).shape == (4,)
    assert select_convergence_indices_by_stratum(
        graded,
        {"graded_core_1": 2, "graded_core_2": 2},
    ).shape == (4,)


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


def test_reference_summary_supports_five_resolutions_and_excludes_band_one() -> None:
    reference = np.ones((2, 81, 10), dtype=np.float64)
    arrays = {resolution: reference.copy() for resolution in (32, 48, 64, 96, 128)}
    arrays[32][..., 0] = 1000.0
    arrays[64][..., 1:] *= 1.001

    summary = compute_reference_convergence_summary(
        arrays,
        reference_resolution=128,
        median_max=0.0011,
        combined_p95_max=0.005,
        lane_p95_max=0.0075,
        scope="combined",
    )

    assert list(summary["comparisons"]) == ["32", "48", "64", "96"]
    assert summary["comparisons"]["32"]["max_mre"] == pytest.approx(0.0)
    assert summary["comparisons"]["64"]["median_mre"] == pytest.approx(0.001)
    assert summary["bands_in_scope"] == list(range(2, 11))


def test_lane_candidate_gate_uses_lane_p95_without_combined_median_rule() -> None:
    reference = np.ones((1, 2, 3), dtype=np.float64)
    arrays = {64: reference * 1.002, 96: reference * 1.002, 128: reference}

    summary = compute_reference_convergence_summary(
        arrays,
        reference_resolution=128,
        median_max=0.001,
        combined_p95_max=0.005,
        lane_p95_max=0.0075,
        scope="binary_air_holes",
    )

    assert summary["comparisons"]["64"]["median_mre"] > 0.001
    assert summary["comparisons"]["64"]["passed_candidate_gate"] is True


def _write_completed_lane(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(
        json.dumps({"status": "pilot_generated_pending_convergence"}),
        encoding="utf-8",
    )
    (run_dir / "dataset_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "status": "pilot_generated_pending_convergence",
                "dataset_path": "data/lane.h5",
                "dataset_sha256": "abc",
            }
        ),
        encoding="utf-8",
    )


def test_finalize_waiter_runs_convergence_and_writes_gated_collection(tmp_path: Path) -> None:
    graded = tmp_path / "graded_run"
    binary = tmp_path / "binary_run"
    convergence = tmp_path / "convergence_run"
    _write_completed_lane(graded)
    _write_completed_lane(binary)
    convergence.mkdir()
    (convergence / "convergence_metrics.json").write_text(
        json.dumps({"status": "resolution_convergence_pass"}),
        encoding="utf-8",
    )
    calls: list[Path] = []

    def fake_convergence_runner(config_path: Path) -> Path:
        calls.append(config_path)
        return convergence

    combined = tmp_path / "combined.yaml"
    result = await_full_cell_pilots_and_finalize(
        graded_run_dir=graded,
        binary_run_dir=binary,
        convergence_config=tmp_path / "convergence.yaml",
        combined_manifest=combined,
        poll_interval_s=0.0,
        timeout_s=1.0,
        convergence_runner=fake_convergence_runner,
    )

    assert result == convergence
    assert len(calls) == 1
    assert yaml.safe_load(combined.read_text(encoding="utf-8"))["status"] == "pilot_pass"
    assert json.loads((convergence / "finalization.json").read_text(encoding="utf-8"))["status"] == "pilot_pass"


def test_finalize_waiter_stops_on_nonpassing_lane(tmp_path: Path) -> None:
    graded = tmp_path / "graded_run"
    binary = tmp_path / "binary_run"
    _write_completed_lane(graded)
    _write_completed_lane(binary)
    (binary / "results.json").write_text(json.dumps({"status": "solver_execution_failed"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="binary_air_holes"):
        await_full_cell_pilots_and_finalize(
            graded_run_dir=graded,
            binary_run_dir=binary,
            convergence_config=tmp_path / "convergence.yaml",
            combined_manifest=tmp_path / "combined.yaml",
            poll_interval_s=0.0,
            timeout_s=1.0,
        )
