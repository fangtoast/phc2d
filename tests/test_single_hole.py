"""Check analytic area, periodicity, shape limits and balanced test coverage.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import collections
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.data.full_cell_generation import (
    load_full_cell_generation_config,
    prepare_full_cell_sample_plan,
)
from phc2d.data.single_hole import build_single_hole_test_plan, evaluate_single_hole
from phc2d.data.full_cell_spec import FullCellStructureSpec


@pytest.mark.parametrize("shape", ["circle", "ellipse", "triangle", "square", "rectangle", "hexagon"])
def test_area_and_periodicity(shape):
    axis = (np.arange(1024) + .5) / 1024 - .5
    x, y = np.meshgrid(axis, axis, indexing="ij")
    params = {"shape": shape, "air_fraction": .17, "aspect_ratio": 2 if shape in {"ellipse", "rectangle"} else 1,
              "angle_deg": 17, "center": [.37, -.43]}
    values = evaluate_single_hole(x, y, params)
    assert abs(float(np.mean(1 - values)) - .17) < .001
    assert np.array_equal(values, evaluate_single_hole(x + 1, y - 2, params))
    assert evaluate_single_hole(.37, -.43, params) == 0


def test_plan_balances_families_and_preserves_exact_limiting_geometries():
    rows = build_single_hole_test_plan()
    assert len({r["sample_id"] for r in rows}) == 1200
    assert set(collections.Counter(r["parameters"]["shape"] for r in rows).values()) == {200}
    x, y = np.meshgrid(np.linspace(-.5, .5, 100), np.linspace(-.5, .5, 100))
    base = {"air_fraction": .15, "aspect_ratio": 1}
    assert np.array_equal(evaluate_single_hole(x, y, {**base, "shape": "circle"}), evaluate_single_hole(x, y, {**base, "shape": "ellipse"}))
    assert np.array_equal(evaluate_single_hole(x, y, {**base, "shape": "square"}), evaluate_single_hole(x, y, {**base, "shape": "rectangle"}))
    structure = FullCellStructureSpec("test", "analytic_single_hole_v1", "binary", {**base, "shape": "triangle"}, (), (), ())
    assert structure.rasterize((64, 64)).shape == (64, 64)


def test_exp_dat_007_config_builds_exact_frozen_production_plan(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/research/full_cell_single_hole_exp_dat_007_v1.yaml"
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))

    assert payload["authorization"] == {
        "formal_generation_authorized": True,
        "authorized_by": "user",
        "authorized_date": "2026-10-20",
        "solver_resolution_pilot_status": "waived_by_user",
        "waiver_scope": "generation_only_no_label_accuracy_or_model_claim",
        "automatic_retry_authorized": False,
        "resource_change_authorized": False,
        "model_evaluation_authorized": False,
    }
    assert payload["dataset"]["split"] == {"train": 0, "val": 0, "test": 1200}

    run_dir = tmp_path / "EXP-DAT-007_plan"
    out = prepare_full_cell_sample_plan(config, run_dir=run_dir, workers=1)
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (out / "sample_plan.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert results["status"] == "full_cell_sample_plan_ready"
    assert len(rows) == 1200
    assert rows[0]["sample_id"] == "single_circle_0000"
    assert rows[-1]["sample_id"] == "single_hexagon_0199"
    assert {row["split"] for row in rows} == {"test"}
    assert collections.Counter(row["stratum"] for row in rows) == {
        shape: 200 for shape in ("circle", "ellipse", "triangle", "square", "rectangle", "hexagon")
    }
    expected = build_single_hole_test_plan()
    assert [
        (row["sample_id"], row["parameters"]) for row in rows
    ] == [
        (row["sample_id"], row["parameters"]) for row in expected
    ]

    axis = (np.arange(256) + 0.5) / 256 - 0.5
    x, y = np.meshgrid(axis, axis, indexing="ij")
    area_errors = [
        abs(float(np.mean(1.0 - evaluate_single_hole(x, y, row["parameters"]))) - row["parameters"]["air_fraction"])
        for row in rows
    ]
    assert max(area_errors) <= 0.01


def test_exp_dat_007_bingxing_overlay_preserves_scientific_contract():
    root = Path(__file__).resolve().parents[1]
    xingyi = yaml.safe_load(
        (root / "configs/research/full_cell_single_hole_exp_dat_007_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    bingxing = yaml.safe_load(
        (
            root
            / "configs/research/full_cell_single_hole_exp_dat_007_bingxing_v1.yaml"
        ).read_text(encoding="utf-8")
    )

    for key in ("experiment", "solver", "k_grid", "acceptance"):
        assert bingxing[key] == xingyi[key]
    for key in ("protocol", "lane", "sample_count", "split", "seed", "raster_resolutions", "raster_contract"):
        assert bingxing["dataset"][key] == xingyi["dataset"][key]
    for key in (
        "algorithm_version",
        "parameter_plan",
        "families",
        "samples_per_family",
        "air_fraction_range",
        "aspect_ratios",
        "epsilon_air",
        "epsilon_material",
        "audit_resolution",
    ):
        assert bingxing["generator"][key] == xingyi["generator"][key]
    assert bingxing["workers"]["workers"] == 256
    assert bingxing["workers"]["max_workers"] == 256
    assert bingxing["authorization"]["resource_change_authorized"] is True


def test_exp_dat_007_bingxing_12k_plan_is_balanced_and_deterministic():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "configs/research/full_cell_single_hole_exp_dat_007_bingxing_12k_v1.yaml"
    )
    payload = load_full_cell_generation_config(path)

    assert payload["dataset"]["sample_count"] == 12_000
    assert payload["dataset"]["split"] == {"train": 0, "val": 0, "test": 12_000}
    assert payload["generator"]["samples_per_family"] == 2_000
    assert payload["workers"]["workers"] == 64
    assert payload["authorization"]["sample_expansion_authorized"] is True

    rows = build_single_hole_test_plan(samples_per_family=2_000)
    assert len(rows) == 12_000
    assert len({row["sample_id"] for row in rows}) == 12_000
    assert len(
        {json.dumps(row["parameters"], sort_keys=True) for row in rows}
    ) == 12_000
    assert collections.Counter(row["parameters"]["shape"] for row in rows) == {
        shape: 2_000
        for shape in ("circle", "ellipse", "triangle", "square", "rectangle", "hexagon")
    }
    assert rows[0]["sample_id"] == "single_circle_0000"
    assert rows[-1]["sample_id"] == "single_hexagon_1999"

    by_shape = {
        shape: [row["parameters"] for row in rows if row["parameters"]["shape"] == shape]
        for shape in ("circle", "ellipse", "triangle", "square", "rectangle", "hexagon")
    }
    assert len({row["air_fraction"] for row in by_shape["circle"]}) == 2_000
    for shape in ("ellipse", "rectangle"):
        assert len({row["air_fraction"] for row in by_shape[shape]}) == 100
        assert {row["aspect_ratio"] for row in by_shape[shape]} == {1.25, 1.5, 2.0, 3.0}
        assert len({row["angle_deg"] for row in by_shape[shape]}) == 5
    for shape in ("triangle", "square", "hexagon"):
        assert len({row["air_fraction"] for row in by_shape[shape]}) == 200
        assert len({row["angle_deg"] for row in by_shape[shape]}) == 10


@pytest.mark.parametrize("shape", ["circle", "ellipse", "triangle", "square", "rectangle", "hexagon"])
def test_single_hole_structure_extraction_preserves_identity_and_raster(shape):
    from phc2d.data.single_hole import build_single_hole_structure
    from phc2d.data.full_cell_generation import _build_single_hole_structure

    ratio = 1.5 if shape in {"ellipse", "rectangle"} else 1.0
    parameters = {"shape": shape, "air_fraction": .17, "aspect_ratio": ratio,
                  "angle_deg": 11.0, "center": [0.0, 0.0]}
    names = ("air_fraction", "aspect_ratio", "angle_deg")
    expected = FullCellStructureSpec(
        sample_id="single_fixture", generator_family="analytic_single_hole_v1",
        material_kind="binary", parameters=parameters, parameter_names=names,
        parameter_vector=(.17, ratio, 11.0), parameter_mask=(True, True, True),
        epsilon_air=1.0, epsilon_material=8.9, metadata={"resumed_from_sample_plan": True},
    )
    actual = build_single_hole_structure(
        sample_id="single_fixture", parameters=parameters, epsilon_air=1.0,
        epsilon_material=8.9, metadata={"resumed_from_sample_plan": True},
    )
    assert _build_single_hole_structure is build_single_hole_structure
    assert actual.geometry_hash == expected.geometry_hash
    assert actual.parameter_hash == expected.parameter_hash
    assert actual.parameter_names == expected.parameter_names
    assert actual.parameter_vector == expected.parameter_vector
    assert actual.schema_version == expected.schema_version
    assert actual.metadata == expected.metadata
    assert np.array_equal(actual.rasterize((32, 32)), expected.rasterize((32, 32)))
