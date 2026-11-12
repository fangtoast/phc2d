"""Verify frozen-checkpoint validation replay boundaries.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from phc2d.inverse import checkpoint_validation_replay as module


def test_validation_replay_and_split_guards(tmp_path, monkeypatch):
    points = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)

    class FakeAdapter:
        def __init__(self, *args, **kwargs):
            self.model_name = "fake"
            self.selection_split = "validation"
            self.validation_bands_2_10_mre_pct = 0.0
            self.k_points_sha256 = hashlib.sha256(points.tobytes()).hexdigest()
            self.input_shape, self.output_shape = (2,), (2, 2)
            self.mean = torch.zeros(2, 2)
            self.std = torch.ones(2, 2)
            self.model = torch.nn.Identity()
            self.spec = None
            self.k_points = torch.from_numpy(points)
            self.lane_index = None
            self.device = torch.device("cpu")

        def predict_normalized_tensor(self, inputs):
            return inputs[:, None, :].expand(-1, 2, -1)

    monkeypatch.setattr(module, "FrozenBandAdapter", FakeAdapter)
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as handle:
        handle["grid"] = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        handle["Wh"] = np.array([[[1, 2], [1, 2]], [[3, 4], [3, 4]],
                                  [[np.nan, np.nan], [np.nan, np.nan]]], dtype=np.float32)
        handle["k_points"] = points
        handle["idx_val"] = [1, 0]
        handle["idx_test"] = [2]

    def run(output):
        return module.replay(dataset=path, dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             grid_field="grid", checkpoint="unused", checkpoint_sha256="unused",
                             expected_model="fake", tolerance_pct_points=0.001,
                             batch_size=1, device="cpu", output=output)

    output = run(tmp_path / "result.json")
    result = json.loads(output.read_text())
    assert result["status"] == "validation_replay_pass"
    assert result["samples"] == 2
    assert result["scope"] == "validation_only_no_test_targets"
    with pytest.raises(FileExistsError):
        run(output)
    FakeAdapter.validation_bands_2_10_mre_pct = 1.0
    original_init = FakeAdapter.__init__

    def mismatched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.validation_bands_2_10_mre_pct = 1.0

    monkeypatch.setattr(FakeAdapter, "__init__", mismatched_init)
    mismatch = json.loads(run(tmp_path / "mismatch.json").read_text())
    assert mismatch["status"] == "validation_replay_failed"
    with h5py.File(path, "r+") as handle:
        handle["idx_test"][0] = 1
    with pytest.raises(ValueError, match="split identity"):
        run(tmp_path / "invalid.json")
