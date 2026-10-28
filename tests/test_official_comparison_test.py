"""Contract checks for frozen official comparison tests.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import json

import numpy as np
import pytest

from phc2d.forward.budget_confirmation import file_digest
from phc2d.forward.official_comparison_test import ready_checkpoint, test_indices as read_test_indices


def test_checkpoint_requires_terminal_marker_and_matching_hash(tmp_path):
    run = tmp_path / "training"
    run.mkdir()
    (run / "model.pt").write_bytes(b"frozen weights")
    digest = file_digest(run / "model.pt")
    result = dict(status="formal_training_success", selection_split="validation",
                  checkpoint_role="best_validation", test_executed=False,
                  checkpoint_sha256=digest)
    (run / "results.json").write_text(json.dumps(result), encoding="utf-8")
    marker = tmp_path / "completed.json"
    marker.write_text(json.dumps(dict(checkpoint_sha256=digest)), encoding="utf-8")
    assert ready_checkpoint(run, marker)["checkpoint_sha256"] == digest
    result["status"] = "training_running"
    (run / "results.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal"):
        ready_checkpoint(run, marker)
    result["status"] = "training_complete_convergence_not_met"
    (run / "results.json").write_text(json.dumps(result), encoding="utf-8")
    marker.write_text(json.dumps(dict(checkpoint_sha256="0" * 64)), encoding="utf-8")
    with pytest.raises(ValueError, match="hashes"):
        ready_checkpoint(run, marker)


def test_test_split_rejects_overlap_and_external_uses_all_without_split(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as handle:
        handle["Wh"] = np.zeros((5, 3, 2), dtype=np.float32)
        handle["idx_train"] = np.array([0, 1])
        handle["idx_val"] = np.array([2])
        handle["idx_test"] = np.array([3, 4])
        assert read_test_indices(handle, external=False).tolist() == [3, 4]
        handle["idx_test"][...] = [1, 4]
        with pytest.raises(ValueError, match="overlaps"):
            read_test_indices(handle, external=False)
        del handle["idx_test"]
        with pytest.raises(ValueError, match="no frozen"):
            read_test_indices(handle, external=False)
        del handle["idx_train"]
        del handle["idx_val"]
        assert len(read_test_indices(handle, external=True)) == 5
