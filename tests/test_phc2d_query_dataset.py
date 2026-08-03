"""Tests for lazy band-query dataset construction.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from phc2d.data import (
    ForwardDatasetBundle,
    BandQueryDataset,
    build_query_masks,
    recover_k_path,
    save_forward_dataset,
    validate_query_split_integrity,
)


def test_band_query_dataset_preserves_lineage_and_lazy_shape(tmp_path):
    bundle = ForwardDatasetBundle(
        rho=np.zeros((5, 36), dtype=np.float32),
        Wh=np.ones((5, 4, 3), dtype=np.float32),
        idx_train=np.array([0, 1, 2]),
        idx_val=np.array([3]),
        idx_test=np.array([4]),
        config_yaml="k_path:\n  n_total: 4\n",
    )
    path = tmp_path / "query.npz"
    loaded = bundle
    save_forward_dataset(path, bundle)

    k_path = recover_k_path(loaded)
    protocols = build_query_masks(loaded.nk, has_segment_metadata=k_path.has_segment_metadata)
    dataset = BandQueryDataset(
        loaded,
        sample_indices=loaded.idx_train,
        k_indices=protocols["unseen_structure_all_k"].train_k_indices,
        k_path=k_path,
        split="train",
        protocol="unseen_structure_all_k",
    )

    row = dataset[0]
    assert len(dataset) == 12
    assert row["structure"].ndim == 2
    assert row["target"].shape == (3,)
    assert row["query_id"].endswith(":k_0000")
    assert k_path.is_physical is False
    assert k_path.source == "normalized_index_from_config_yaml.k_path.n_total"


def test_query_masks_keep_structure_and_heldout_k_disjoint():
    bundle = ForwardDatasetBundle(
        rho=np.zeros((6, 36), dtype=np.float32),
        Wh=np.ones((6, 8, 2), dtype=np.float32),
        idx_train=np.array([0, 1, 2]),
        idx_val=np.array([3, 4]),
        idx_test=np.array([5]),
    )
    k_path = recover_k_path(bundle)
    protocols = build_query_masks(bundle.nk, has_segment_metadata=k_path.has_segment_metadata)
    integrity = validate_query_split_integrity(bundle, protocols)

    assert integrity["train_test_sample_id_disjoint"] is True
    assert integrity["protocols"]["unseen_structure_k_interpolation"]["heldout_k_absent_from_train"] is True
    assert integrity["protocols"]["unseen_structure_heldout_segment"]["holdout_kind"] == "contiguous_index_holdout"
