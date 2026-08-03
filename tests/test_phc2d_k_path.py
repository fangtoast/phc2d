"""Tests for versioned physical k-path sidecars and query leakage rules.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import numpy as np
import yaml

from phc2d.data import (
    ForwardDatasetBundle,
    BandQueryDataset,
    build_physical_query_masks,
    build_square_gxm_reduced_spec,
    detect_duplicate_coordinate_groups,
    discover_k_path_sidecar,
    expand_physical_holdout_indices,
    hash_k_path_spec,
    infer_dataset_schema,
    load_forward_dataset,
    load_k_path_spec,
    physical_k_ids,
    recover_k_path,
    save_forward_dataset,
    save_k_path_spec,
    validate_k_path_spec,
)


def test_k_path_spec_round_trip_and_hash_stability(tmp_path):
    spec = build_square_gxm_reduced_spec(n_total=31)
    assert validate_k_path_spec(spec, expected_nk=31) == []
    assert spec.nodes[1].label == "X"
    assert spec.node_indices["X"] == 10
    assert spec.node_indices["M"] == 20
    assert spec.duplicate_coordinate_groups == ((0, 30),)
    assert spec.path_s[0] == 0.0
    assert spec.path_s[-1] > spec.path_s[20]

    path = tmp_path / "square.kpath.v1.json"
    save_k_path_spec(spec, path)
    loaded = load_k_path_spec(path)
    assert loaded.k_points == spec.k_points
    assert loaded.k_points_hash == spec.k_points_hash
    assert hash_k_path_spec(loaded) == hash_k_path_spec(spec)


def test_duplicate_physical_k_id_and_holdout_expansion():
    points = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.0]], dtype=np.float32)
    assert detect_duplicate_coordinate_groups(points) == ((0, 2),)
    ids = physical_k_ids(points)
    assert ids[0] == ids[2]
    assert expand_physical_holdout_indices([2], ids) == (0, 2)


def test_sidecar_discovery_schema_and_query_dataset_metadata(tmp_path):
    data_path = tmp_path / "toy.npz"
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    bundle = ForwardDatasetBundle(
        rho=np.zeros((3, 36), dtype=np.float32),
        Wh=np.ones((3, 31, 2), dtype=np.float32),
        idx_train=np.array([0], dtype=np.int64),
        idx_val=np.array([1], dtype=np.int64),
        idx_test=np.array([2], dtype=np.int64),
    )
    save_forward_dataset(data_path, bundle)
    spec = build_square_gxm_reduced_spec(n_total=31)
    sidecar = metadata_dir / "square.kpath.v1.json"
    save_k_path_spec(spec, sidecar)
    manifest = {
        "dataset_path": str(data_path),
        "k_path_sidecar": sidecar.name,
        "k_path_hash": hash_k_path_spec(spec),
    }
    (metadata_dir / "toy.manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")

    assert discover_k_path_sidecar(data_path) == sidecar.resolve()
    schema = infer_dataset_schema(data_path)
    assert schema["physical_k_coordinates_available"] is True
    assert schema["k_path_verified"] is False
    assert schema["coordinate_system"] == "mpb_reduced_reciprocal_coordinates"

    loaded = load_forward_dataset(data_path)
    k_path = recover_k_path(loaded)
    protocols = build_physical_query_masks(k_path)
    q2 = protocols["unseen_structure_k_interpolation"]
    assert q2.holdout_kind == "physical_coordinate_interpolation"
    assert not (set(q2.train_k_indices) & set(q2.eval_k_indices))

    dataset = BandQueryDataset(
        loaded,
        sample_indices=loaded.idx_train,
        k_indices=(0, 30),
        k_path=k_path,
        split="train",
        protocol="unit",
    )
    first = dataset[0]
    last = dataset[1]
    assert first["k_coordinate"].shape == (2,)
    assert first["physical_k_id"] == last["physical_k_id"]
    assert first["query_id"].split(":")[1] == spec.k_path_id
    assert first["path_s"].shape == ()
