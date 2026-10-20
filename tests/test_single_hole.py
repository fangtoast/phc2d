"""Check analytic area, periodicity, shape limits and balanced test coverage.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import collections
import numpy as np
import pytest

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
