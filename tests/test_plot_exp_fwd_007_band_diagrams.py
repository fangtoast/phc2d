"""Tests for EXP-FWD-007 representative band-diagram selection.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from scripts import plot_exp_fwd_007_band_diagrams as plotter


def test_complete_gaps_use_full_bz_extrema() -> None:
    target = np.asarray(
        [
            [[0.10, 0.30, 0.50], [0.20, 0.32, 0.56]],
            [[0.15, 0.31, 0.52], [0.18, 0.33, 0.58]],
        ]
    ).reshape(4, 3)

    gaps = plotter._complete_gaps(target)

    assert gaps == [(1, 0.2, 0.3), (2, 0.33, 0.5)]


def test_gap_sample_selection_returns_low_median_and_high_error() -> None:
    target = np.zeros((5, 2, 3), dtype=float)
    target[:, :, 0] = 0.1
    target[:, :, 1] = 0.2
    target[:, :, 2] = 0.3
    target[1, 1, 0] = 0.25
    target[1, 1, 1] = 0.35
    target[3, 1, 0] = 0.25
    target[3, 1, 1] = 0.35
    sample_mre = np.asarray([0.7, 0.1, 0.5, 0.2, 0.9])

    selected = plotter._select_gap_samples(target, sample_mre)

    assert selected == [2, 0, 4]


def test_full_bz_rows_keep_each_fixed_kx_group_separate() -> None:
    values = np.linspace(-4 / 9, 4 / 9, 9)
    k_points = np.asarray([(k_x, k_y) for k_x in values for k_y in values])

    rows = plotter._full_bz_row_slices(k_points)

    assert [(row.start, row.stop) for row in rows] == [
        (0, 9),
        (9, 18),
        (18, 27),
        (27, 36),
        (36, 45),
        (45, 54),
        (54, 63),
        (63, 72),
        (72, 81),
    ]
