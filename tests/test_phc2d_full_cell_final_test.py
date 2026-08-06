"""Tests for final full-cell checkpoint routing and frozen statistics.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import numpy as np

from phc2d.forward.full_cell_final_test import _bootstrap_mean_ci, resolve_final_run_dirs


def test_resolve_final_run_dirs_routes_main_and_ablation_checkpoints() -> None:
    final = [
        {"representation": "binary", "model": "fno2d_band_v2", "resolution": "64"},
        {"representation": "continuous", "model": "res_mlp_wide", "resolution": "32"},
        {"representation": "t05", "model": "lifted_cnn", "resolution": "128"},
    ]
    main = [
        {"wave": "binary", "model": "fno2d_band_v2", "seed": str(seed), "run_dir": f"binary_s{seed}"}
        for seed in (0, 1, 2)
    ]
    ablation = [
        {"representation": representation, "model": model, "resolution": str(resolution), "seed": str(seed), "run_dir": f"{representation}_r{resolution}_s{seed}"}
        for representation, model, resolution in (("continuous", "res_mlp_wide", 32), ("t05", "lifted_cnn", 128))
        for seed in (0, 1, 2)
    ]
    resolved = resolve_final_run_dirs(final, main, ablation)
    assert len(resolved) == 9
    assert resolved[("binary", 2)] == "binary_s2"
    assert resolved[("continuous", 1)] == "continuous_r32_s1"
    assert resolved[("t05", 0)] == "t05_r128_s0"


def test_bootstrap_mean_ci_uses_frozen_resample_indices() -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    indices = np.asarray([[0, 0, 0], [0, 1, 2], [2, 2, 2]])

    estimate, low, high = _bootstrap_mean_ci(values, indices=indices)

    assert estimate == 2.0
    assert low == 1.05
    assert high == 2.95
