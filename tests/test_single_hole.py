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

from phc2d.data.full_cell_generation import prepare_full_cell_sample_plan
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
