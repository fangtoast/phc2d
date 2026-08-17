"""Architecture and compatibility gates for reusable full-cell interfaces.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from phc2d.data.full_cell import FullCellStructureSpec as FacadeStructureSpec
from phc2d.data.full_cell_fourier import (
    evaluate_real_fourier_field,
    periodic_cell_centers,
)
from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.generators.full_cell import (
    generate_binary_polar_holes as facade_binary_polar,
)
from phc2d.generators.full_cell_binary import generate_binary_polar_holes


ROOT = Path(__file__).resolve().parents[1]


def test_full_cell_facade_preserves_public_object_identity() -> None:
    assert FacadeStructureSpec is FullCellStructureSpec
    assert (
        importlib.import_module("phc2d.data").FullCellStructureSpec
        is FullCellStructureSpec
    )
    assert facade_binary_polar is generate_binary_polar_holes


def test_full_cell_data_facade_is_declarative_only() -> None:
    source = (ROOT / "src/phc2d/data/full_cell.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert "__module__" not in source
    generator_source = (ROOT / "src/phc2d/generators/full_cell.py").read_text(
        encoding="utf-8"
    )
    generator_tree = ast.parse(generator_source)
    assert not [
        node
        for node in generator_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def test_fourier_and_spec_modules_have_no_heavy_imports() -> None:
    forbidden = {"h5py", "scipy", "torch", "meep", "matplotlib"}
    for relative in (
        "src/phc2d/data/full_cell_fourier.py",
        "src/phc2d/data/full_cell_spec.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            str(node.module).split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        assert imported.isdisjoint(forbidden)


def test_fourier_field_matches_hand_computed_modes_and_broadcasts() -> None:
    x = np.asarray([0.0, 0.25], dtype=np.float64)
    y = np.asarray([[0.0], [0.5]], dtype=np.float64)
    modes = [
        {"mx": 1, "my": 0, "cos": 2.0, "sin": -0.5},
        {"mx": 0, "my": 1, "cos": 0.25},
    ]
    expected = (
        2.0 * np.cos(2.0 * np.pi * x)
        - 0.5 * np.sin(2.0 * np.pi * x)
        + 0.25 * np.cos(2.0 * np.pi * y)
    )
    actual = evaluate_real_fourier_field(x, y, modes)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64


def test_periodic_cell_centers_keep_legacy_coordinate_contract() -> None:
    x, y = periodic_cell_centers((2, 4))
    np.testing.assert_array_equal(x[0], [-0.375, -0.125, 0.125, 0.375])
    np.testing.assert_array_equal(y[:, 0], [-0.25, 0.25])


def test_lightweight_import_does_not_load_optional_heavy_modules() -> None:
    code = """
import json, sys
import phc2d.data.full_cell_fourier
import phc2d.data.full_cell_spec
print(json.dumps(sorted(set(sys.modules) & {'h5py','scipy','torch','meep','matplotlib'})))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout.strip()) == []
