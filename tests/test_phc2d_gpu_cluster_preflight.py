"""Tests for the full-cell GPU-cluster preflight contract.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from phc2d.runs import validate_run_artifacts
from phc2d.training.gpu_cluster_preflight import run_gpu_cluster_preflight


def test_gpu_cluster_preflight_writes_complete_cpu_test_artifacts(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional and not installed")
    h5py = pytest.importorskip("h5py")
    dataset = tmp_path / "tiny_full_cell.h5"
    with h5py.File(dataset, "w") as handle:
        handle.attrs["schema_version"] = "phc2d.full_cell.v1"
        handle.create_dataset(
            "material_weight_grid_8",
            data=np.random.default_rng(7).random((8, 8, 8), dtype=np.float32),
        )
        handle.create_dataset("Wh", data=np.random.default_rng(8).random((8, 3, 2), dtype=np.float32))
        handle.create_dataset("idx_train", data=np.arange(6, dtype=np.int64))
        handle.create_dataset("idx_val", data=np.asarray([6], dtype=np.int64))
        handle.create_dataset("idx_test", data=np.asarray([7], dtype=np.int64))

    config = tmp_path / "preflight.yaml"
    config.write_text(
        "\n".join(
            [
                "experiment:",
                "  experiment_id: unit_gpu_preflight",
                "dataset:",
                f"  path: {dataset.as_posix()}",
                "  grid_field: material_weight_grid_8",
                "  preflight_samples: 6",
                "  throughput_samples: 6",
                "model:",
                "  candidates: [mlp, res_mlp_wide, lifted_cnn, fno2d_band_v2]",
                "  overrides:",
                "    mlp: {hidden_dim: 8, depth: 1}",
                "    res_mlp_wide: {hidden_dim: 8, depth: 1}",
                "    lifted_cnn: {channels: 4}",
                "    fno2d_band_v2: {width: 4, depth: 1, modes: [2, 2], band_modes: [2, 2], fourier_rank: 2}",
                "training:",
                "  batch_size: 2",
                "  fno_epochs: 1",
                "runtime:",
                "  require_cuda: false",
                "  hdf5_num_workers: 0",
                "output:",
                f"  runs_dir: {(tmp_path / 'runs').as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = run_gpu_cluster_preflight(config)

    assert validate_run_artifacts(run_dir) == []
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert results["status"] == "gpu_single_card_preflight_pass"
    assert results["formal_training_started"] is False
    assert {row["model"] for row in results["model_checks"]} == {
        "mlp",
        "res_mlp_wide",
        "lifted_cnn",
        "fno2d_band_v2",
    }
    assert metrics["fno_preflight_epochs"] == 1
    assert metrics["checkpoint_reload_exact_match"] is True
