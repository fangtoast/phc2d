"""Tests for unified-Fourier full-cell diagnostics.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from pathlib import Path

import numpy as np
import pytest

from phc2d.plot.full_cell_fourier import plot_full_cell_fourier_dataset


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
