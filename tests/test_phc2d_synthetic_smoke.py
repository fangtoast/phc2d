"""Tests for deterministic non-physical clean-clone smoke datasets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.data import SYNTHETIC_SMOKE_STATUS, load_forward_dataset
from phc2d.data.synthetic_smoke import make_synthetic_smoke_dataset


def _write_config(path: Path, output: Path, *, seed: int = 19) -> None:
    payload = {
        "schema_version": "phc2d.synthetic_smoke.v1",
        "dataset": {
            "output_path": output.as_posix(),
            "sample_count": 12,
            "rho_wedge_length": 10,
        },
        "target": {"name": "Wh", "k_point_count": 7, "band_count": 3},
        "split": {"train": 8, "val": 2, "test": 2},
        "generation": {"seed": seed, "fill_probability": 0.4},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_synthetic_smoke_uses_config_shapes_and_explicit_status(tmp_path: Path) -> None:
    config = tmp_path / "synthetic.yaml"
    output = tmp_path / "smoke.npz"
    _write_config(config, output)

    result = make_synthetic_smoke_dataset(config)
    bundle = load_forward_dataset(output)
    manifest = yaml.safe_load(output.with_suffix(".manifest.yaml").read_text(encoding="utf-8"))

    assert result["evidence_status"] == SYNTHETIC_SMOKE_STATUS
    assert bundle.rho.shape == (12, 10)
    assert bundle.Wh.shape == (12, 7, 3)
    assert (len(bundle.idx_train), len(bundle.idx_val), len(bundle.idx_test)) == (8, 2, 2)
    assert all(row["evidence_status"] == SYNTHETIC_SMOKE_STATUS for row in bundle.metadata)
    assert manifest["scientific_claims_allowed"] is False
    assert manifest["physical_verification_status"] == "not_applicable"


def test_synthetic_smoke_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    first_config = tmp_path / "first.yaml"
    second_config = tmp_path / "second.yaml"
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_config(first_config, first)
    _write_config(second_config, second)

    make_synthetic_smoke_dataset(first_config)
    make_synthetic_smoke_dataset(second_config)
    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        np.testing.assert_array_equal(left["rho"], right["rho"])
        np.testing.assert_array_equal(left["Wh"], right["Wh"])

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        make_synthetic_smoke_dataset(first_config)
    make_synthetic_smoke_dataset(first_config, overwrite=True)


def test_synthetic_smoke_rejects_inconsistent_split(tmp_path: Path) -> None:
    config = tmp_path / "synthetic.yaml"
    output = tmp_path / "smoke.npz"
    _write_config(config, output)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["split"]["test"] = 3
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="must equal"):
        make_synthetic_smoke_dataset(config)
