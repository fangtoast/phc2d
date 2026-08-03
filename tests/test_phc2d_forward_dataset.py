"""Tests for the shared forward dataset contract."""

from __future__ import annotations

import numpy as np

from phc2d.data import ForwardDatasetBundle, load_forward_dataset, save_forward_dataset


def test_forward_dataset_roundtrip(tmp_path):
    bundle = ForwardDatasetBundle(
        rho=np.zeros((4, 36), dtype=np.float32),
        Wh=np.ones((4, 3, 2), dtype=np.float32),
        idx_train=np.array([0, 1]),
        idx_val=np.array([2]),
        idx_test=np.array([3]),
        config_yaml="dataset:\n  backend: fem\n",
        metadata=[{"generator_name": "legacy_random", "seed": 1}],
    )
    out = tmp_path / "tiny.npz"

    save_forward_dataset(out, bundle)
    loaded = load_forward_dataset(out)

    assert loaded.n_samples == 4
    assert loaded.nk == 3
    assert loaded.nb == 2
    assert loaded.idx_train.tolist() == [0, 1]
    assert loaded.metadata[0]["generator_name"] == "legacy_random"
    assert loaded.metadata[0]["sample_id"] == "tiny_000000"
    assert loaded.metadata[0]["sample_id_source"] == "derived_from_index"
    assert loaded.metadata[0]["split"] == "train"
    assert "backend" in loaded.config_yaml

