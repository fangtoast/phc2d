"""Tests for shared plotting API."""

from __future__ import annotations

import numpy as np

from phc2d.plot import plot_band_overlay, plot_field_map, plot_medium_map, plot_metric_report


def test_plot_api_writes_pngs(tmp_path):
    rho = np.zeros(36, dtype=np.float32)
    rho[::2] = 1.0

    medium = plot_medium_map(rho, tmp_path / "medium.png")
    bands = plot_band_overlay(np.ones((3, 2)), tmp_path / "bands.png")
    metrics = plot_metric_report({"mse": 0.1, "gap_f1": 0.5}, tmp_path / "metrics.png")
    field = plot_field_map(np.ones((4, 4)), tmp_path / "field.png")

    for result in [medium, bands, metrics, field]:
        assert result.path.is_file()
        assert result.path.stat().st_size > 0
