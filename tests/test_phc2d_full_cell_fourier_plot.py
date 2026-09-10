"""Tests for unified-Fourier full-cell diagnostics.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from phc2d.plot.full_cell_fourier import plot_full_cell_fourier_dataset
from phc2d.plot.fourier_cutoff_study import plot_fourier_cutoff_study


def test_plot_full_cell_fourier_dataset(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    source = tmp_path / "pilot.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("sample_id", data=np.asarray([b"s0", b"s1"]))
        handle.create_dataset("material_weight_grid_8", data=np.asarray([np.zeros((8, 8)), np.ones((8, 8))]))
        base = np.linspace(0.1, 1.0, 10)
        handle.create_dataset("Wh", data=np.stack([np.tile(base, (9, 1)), np.tile(base + 0.01, (9, 1))]))
    summary = plot_full_cell_fourier_dataset(source, tmp_path / "figures")
    assert summary["sample_count"] == 2
    assert summary["finite"] and summary["ascending"]
    assert (tmp_path / "figures/sample_atlas.png").is_file()
    assert (tmp_path / "figures/distribution_and_bands.svg").is_file()
    svg = (tmp_path / "figures/distribution_and_bands.svg").read_text(encoding="utf-8")
    assert "#10b981" not in svg and "#ef4444" not in svg


def test_plot_fourier_cutoff_study_writes_blue_white_outputs(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    for lane in ("binary", "graded"):
        for cutoff in range(5, 21):
            member = tmp_path / f"{lane}_k{cutoff:02d}"
            member.mkdir()
            grid = np.full((1, 8, 8), 0.7 if lane == "binary" else 0.69)
            structure = {
                "parameters": {"modes": [{"mx": 1, "my": 0}] * cutoff},
                "metadata": {"stats": {"boundary_variation_per_a": cutoff, "spectral_second_moment": cutoff**2}},
            }
            with h5py.File(member / "dataset.h5", "w") as handle:
                handle.create_dataset("structure_json", data=np.asarray([json.dumps(structure).encode()]))
                handle.create_dataset(f"material_{'binary' if lane == 'binary' else 'weight'}_grid_256", data=grid)
                handle.create_dataset("Wh", data=np.full((1, 81, 10), 0.1 + cutoff / 100.0))
    result = plot_fourier_cutoff_study(tmp_path, tmp_path / "figures")
    assert result["member_count"] == 32
    assert (tmp_path / "figures/paired_structures_k05_k20.png").is_file()
    assert (tmp_path / "figures/binary_structures_k05_k20.png").is_file()
    assert (tmp_path / "figures/graded_structures_k05_k20.png").is_file()
    svg = (tmp_path / "figures/structure_and_band_effects.svg").read_text(encoding="utf-8")
    assert "#ef4444" not in svg and "#10b981" not in svg
