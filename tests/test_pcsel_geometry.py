"""Contracts for the approved planar PCSEL geometry families.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import pytest

from phc2d.data.pcsel_geometry import (
    PCSEL_FAMILIES, build_pcsel_test_plan, certify_pcsel_clearance,
    evaluate_pcsel, pcsel_primitives, pcsel_parameter_rows,
)
from phc2d.data.full_cell_generation import load_full_cell_generation_config, _validate_config
from phc2d.data.full_cell_spec import FullCellStructureSpec


def test_approved_plan_has_exact_counts_and_unique_parameters():
    rows = build_pcsel_test_plan()
    assert Counter(r["parameters"]["shape"] for r in rows) == {f: 2000 for f in PCSEL_FAMILIES}
    assert len({r["sample_id"] for r in rows}) == 24000
    assert len({json.dumps(r["parameters"], sort_keys=True) for r in rows}) == 24000
    assert rows == build_pcsel_test_plan()


@pytest.mark.parametrize("family", PCSEL_FAMILIES)
def test_periodic_area_scalar_and_vector_evaluators_agree(family):
    axis = (np.arange(512)+.5)/512-.5
    x, y = np.meshgrid(axis, axis)
    for i in [0, 1999]:
        p = pcsel_parameter_rows(family)[i]
        parameters = dict(shape=family, air_fraction=p["f"], primitives=pcsel_primitives(family, p))
        material = evaluate_pcsel(x, y, parameters)
        assert abs(float(np.mean(1-material))-p["f"]) < .002
        assert np.array_equal(material, evaluate_pcsel(x+1, y-2, parameters))
        for a, b in [(.12, .27), (-.39, -.31), (.45, .02)]:
            assert evaluate_pcsel(a, b, parameters) == evaluate_pcsel(np.array([a]), np.array([b]), parameters)[0]
        assert certify_pcsel_clearance(parameters) > 0
        spec = FullCellStructureSpec("probe", "analytic_pcsel_v1", "binary", parameters, (), (), ())
        assert spec.epsilon_at(.12, .27) in [1., 8.9]


def test_right_triangle_and_noda_iii_anchors():
    parts = pcsel_primitives("S02", dict(f=.16, theta=0))
    vertices = np.array(parts[0]["vertices"])
    edges = vertices[1:]-vertices[0]
    assert np.isclose(np.dot(*edges), 0)
    assert np.isclose(np.linalg.norm(edges[0]), np.linalg.norm(edges[1]))
    p = pcsel_primitives("D03", dict(f=.08, w=9/16, d=.257, c=.45))
    assert np.allclose(np.array(p[1]["center"])-p[0]["center"], [.257, -.257])
    assert math.isclose(math.pi*p[0]["a"]*p[0]["b"], .045)
    assert math.isclose(math.pi*p[1]["a"]*p[1]["b"], .035)


def test_slender_triangle_has_certified_edge_normal_clearance():
    p = pcsel_parameter_rows("S01")[1997]
    params = dict(shape="S01", primitives=pcsel_primitives("S01", p))
    assert certify_pcsel_clearance(params) > .05


def test_config_rejects_changed_counts_or_missing_authorization():
    root = Path(__file__).resolve().parents[1]
    cfg = load_full_cell_generation_config(root/"configs/research/full_cell_pcsel_exp_dat_008_bingxing_v1.yaml")
    assert not _validate_config(cfg)
    bad = json.loads(json.dumps(cfg))
    bad["authorization"]["formal_generation_authorized"] = False
    assert _validate_config(bad)
    bad = json.loads(json.dumps(cfg))
    bad["generator"]["samples_per_family"] = 1000
    assert _validate_config(bad)
