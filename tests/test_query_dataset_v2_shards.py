"""Tests for Query Dataset V2 shard generation and merge helpers.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from phc2d.data.forward_dataset import load_forward_dataset
from scripts.generate_query_dataset_v2_shard import generate_shard
from scripts.merge_query_dataset_v2_shards import merge_shards
from scripts.run_query_dataset_v2_sharded_generation import build_shard_specs


def _base_config(tmp_path: Path) -> dict:
    return {
        "experiment_id": "query_dataset_v2_pilot_unit",
        "dataset": {
            "output_path": str(tmp_path / "query_dataset_v2_pilot_unit.npz"),
            "sample_count": 3,
            "split": {"train": 1, "val": 1, "test": 1},
            "sample_id_policy": "stable_generated_id",
            "seed": 7,
            "structure_generator": {"name": "legacy_random", "prob_high": 0.5},
        },
        "geometry": {
            "representation": "rho_wedge",
            "grid_shape_source": "config",
            "grid_shape": [4, 4],
            "n_wedge": "from_grid_shape",
            "lifting": "p4m_wedge_binary_pixels_as_mpb_blocks_air_background",
        },
        "solver": {
            "backend": "mpb",
            "environment": "wsl_meep_env",
            "use_proxy": False,
            "lattice": "square",
            "lattice_constant": 1.0,
            "epsilon_air": 1.0,
            "epsilon_material": 8.9,
            "resolution": 8,
            "polarization": "TE",
            "num_bands": 2,
            "band_guard_count": 1,
            "band_order_policy": "frequency_ascending",
            "eigensolver_tolerance": 1.0e-6,
            "frequency_normalization": "mpb_dimensionless_a_over_lambda",
        },
        "k_path": {
            "source": "generated_from_KPathSpec",
            "family": "square_gxm",
            "k_path_id": "query_dataset_v2_unit_square_gxm",
            "n_total": 4,
            "coordinate_system": "mpb_reduced_reciprocal_coordinates",
            "endpoint_policy": "explicit",
        },
        "output": {
            "runs_dir": str(tmp_path / "runs"),
            "write_manifest": True,
            "write_k_path_sidecar": True,
            "write_solver_provenance": True,
            "write_metadata_jsonl": True,
        },
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "query_dataset_v2_pilot_unit.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _fake_solver(rho, *, k_points, config):
    fill = float(np.asarray(rho, dtype=np.float32).sum())
    bands = np.full((len(k_points), int(config.num_bands)), fill, dtype=np.float32)
    return bands, {
        "mpb_backend": "fake_unit_solver",
        "num_bands": int(config.num_bands),
        "band_guard_count": int(config.band_guard_count),
        "solved_num_bands": int(config.num_bands) + int(config.band_guard_count),
        "band_order_policy": str(config.band_order_policy),
        "eigensolver_tolerance": float(config.eigensolver_tolerance),
    }


def test_build_shard_specs_covers_total_with_tail() -> None:
    specs = build_shard_specs(total_samples=10, shard_size=4)

    assert [(spec.shard_index, spec.start, spec.end, spec.name) for spec in specs] == [
        (0, 0, 4, "shard_000"),
        (1, 4, 8, "shard_001"),
        (2, 8, 10, "shard_002"),
    ]


def test_generate_shard_writes_only_requested_sample_range(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _base_config(tmp_path))
    shard_path = generate_shard(
        config_path,
        start=1,
        end=3,
        out=tmp_path / "shards" / "shard_001.npz",
        run_dir=tmp_path / "runs" / "shard_001",
        solver_fn=_fake_solver,
    )

    with np.load(shard_path, allow_pickle=False) as data:
        assert data["rho"].shape[0] == 2
        assert data["Wh"].shape[0] == 2
        assert data["sample_index"].tolist() == [1, 2]
        metadata = [json.loads(line) for line in str(data["metadata_jsonl"].item()).splitlines()]

    assert [row["sample_index"] for row in metadata] == [1, 2]
    assert (tmp_path / "runs" / "shard_001" / "worker_shards" / "query_dataset_v2_pilot_unit_000001.json").is_file()


def test_merge_shards_writes_standard_forward_dataset(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _base_config(tmp_path))
    first = generate_shard(
        config_path,
        start=0,
        end=1,
        out=tmp_path / "shards" / "shard_000.npz",
        run_dir=tmp_path / "runs" / "shard_000",
        solver_fn=_fake_solver,
    )
    second = generate_shard(
        config_path,
        start=1,
        end=3,
        out=tmp_path / "shards" / "shard_001.npz",
        run_dir=tmp_path / "runs" / "shard_001",
        solver_fn=_fake_solver,
    )

    out_path = merge_shards(
        config_path,
        shards=[first, second],
        out=tmp_path / "query_dataset_v2_pilot_unit_merged.npz",
        run_dir=tmp_path / "runs" / "merge",
    )

    bundle = load_forward_dataset(out_path)
    manifest = yaml.safe_load((tmp_path / "metadata" / "query_dataset_v2_pilot_unit_merged.manifest.yaml").read_text(encoding="utf-8"))

    assert bundle.rho.shape[0] == 3
    assert bundle.Wh.shape == (3, 4, 2)
    assert bundle.idx_train.tolist() == [0]
    assert bundle.idx_val.tolist() == [1]
    assert bundle.idx_test.tolist() == [2]
    assert [row["sample_index"] for row in bundle.metadata] == [0, 1, 2]
    assert manifest["provenance_status"] == "shards_merged"
