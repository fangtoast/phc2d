"""Independent branch and mixed-split contracts for EXP-FWD-022.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.data.full_cell_mixed import file_sha256, materialize_mixed_validation
from phc2d.forward.global_local_branch_capability import parameter_controls


def source(tmp_path: Path, name: str, overlap: bool = False) -> dict:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / f"{name}.h5"
    with h5py.File(path, "w") as h:
        h["grid"] = np.full((6, 4, 4), 0.3, dtype=np.float32)
        target = np.ones((6, 2, 3), dtype=np.float32)
        target[5] = np.nan  # Test-only labels must never enter mixed data.
        h["Wh"] = target
        h["k_points"] = np.zeros((2, 2), dtype=np.float32)
        h["idx_train"] = np.arange(4)
        h["idx_val"] = np.array([0 if overlap else 4])
        h["idx_test"] = np.array([5])
    return {"path": str(path), "sha256": file_sha256(path), "grid_field": "grid"}


def test_mixed_materialization_preserves_equal_split_lineage_without_test_labels(tmp_path):
    h5py = pytest.importorskip("h5py")
    sources = {x: source(tmp_path, x) for x in ("k5", "k20")}
    output = tmp_path / "mixed.h5"
    result = materialize_mixed_validation(sources, output, train_per_domain=4, validation_per_domain=1)
    assert result["n_train"] == 8 and result["n_val"] == 2
    assert result["test_targets_loaded"] is False
    with h5py.File(output) as h:
        assert np.isfinite(h["Wh"][:]).all()
        assert len(h["idx_test"]) == 0
        assert h["source_domain_index"][:].tolist() == [0] * 4 + [1] * 4 + [0, 1]
        assert h["source_row_index"][:].tolist() == list(range(4)) * 2 + [4, 4]
    with pytest.raises(FileExistsError):
        materialize_mixed_validation(sources, output, train_per_domain=4, validation_per_domain=1)


def test_mixed_materialization_rejects_split_overlap(tmp_path):
    sources = {"k5": source(tmp_path, "k5", overlap=True), "k20": source(tmp_path, "k20")}
    with pytest.raises(ValueError, match="overlap"):
        materialize_mixed_validation(sources, tmp_path / "mixed.h5", train_per_domain=4, validation_per_domain=1)


def test_mixed_materialization_rejects_source_hash_drift(tmp_path):
    sources = {x: source(tmp_path, x) for x in ("k5", "k20")}
    sources["k5"]["sha256"] = "invalid"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        materialize_mixed_validation(sources, tmp_path / "mixed.h5", train_per_domain=4, validation_per_domain=1)


def test_independent_branches_are_physically_removed_and_full_remains_compatible():
    torch = pytest.importorskip("torch")
    from phc2d.models import build_model

    kwargs = dict(input_shape=(4, 4), output_shape=(2, 3), width=8, depth=1,
                  modes=(2, 2), norm_groups=2, head_hidden_dim=8, local_kernel=3)
    torch.manual_seed(9)
    default = build_model("global_local_band_operator_v1", **kwargs)
    torch.manual_seed(9)
    explicit = build_model("global_local_band_operator_v1", **kwargs, enabled_branches=["spectral", "local"])
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert all(torch.equal(x, explicit.state_dict()[k]) for k, x in default.state_dict().items())
    for branch, absent in (("spectral", "local"), ("local", "spectral")):
        single = build_model("global_local_band_operator_v1", **kwargs, enabled_branches=[branch])
        assert getattr(single.blocks[0], absent) is None
        assert all(f"blocks.0.{absent}." not in k for k in single.state_dict())
        prediction = single(torch.rand(2, 1, 4, 4))
        assert prediction.shape == (2, 2, 3)
        prediction.square().mean().backward()


def test_width_only_parameter_gate_fails_closed_for_frozen_global_only_contract():
    pytest.importorskip("torch")
    controller = yaml.safe_load(Path("configs/research/full_cell_global_local_bridge_exp_fwd_020_v1.yaml").read_text())
    params = {**controller["model"]["shared_params"], "local_kernel": 3}
    with pytest.raises(RuntimeError, match="parameter-count control unavailable: g_only"):
        parameter_controls(params, (64, 64), (81, 10), {"min": 8, "max": 512, "step": 8}, 0.05)
