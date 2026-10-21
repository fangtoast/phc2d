"""Check frozen external-evaluation data identity before GPU inference."""

import hashlib
import json

import numpy as np
import pytest

from phc2d.forward.budget_confirmation import file_digest
from phc2d.forward.frozen_checkpoint_evaluation import _external_contract, _indices_and_families, summarize_campaign


def test_external_data_requires_identical_k_order_and_grid_shape(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "external.h5"
    k = np.asarray([[0, 0], [0.5, 0]], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle["material_area_fraction_grid_64"] = np.zeros((2, 4, 4), dtype=np.float32)
        handle["Wh"] = np.ones((2, 2, 3), dtype=np.float32)
        handle["k_points"] = k
        handle["idx_test"] = [0, 1]
        handle["structure_json"] = [json.dumps({"parameters": {"shape": shape}}).encode()
                                    for shape in ("circle", "square")]
    reference = {"grid_shape": (4, 4), "target_shape": (2, 3),
                 "k_points_sha256": hashlib.sha256(k.tobytes()).hexdigest()}
    digest = file_digest(path)
    assert _external_contract(path, digest, reference, "material_area_fraction_grid_64")["sha256"] == digest
    indices, families = _indices_and_families(path, "test")
    assert indices.tolist() == [0, 1] and families == ["circle", "square"]
    with pytest.raises(ValueError, match="SHA-256"):
        _external_contract(path, "incorrect", reference, "material_area_fraction_grid_64")
    reference["k_points_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="k-point order"):
        _external_contract(path, digest, reference, "material_area_fraction_grid_64")


def test_summary_refuses_incomplete_checkpoint_inventory(tmp_path):
    manifest = tmp_path / "test_manifest.csv"
    manifest.write_text("task_id,checkpoint_sha256\none,abc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="54 unique"):
        summarize_campaign(manifest, tmp_path)
