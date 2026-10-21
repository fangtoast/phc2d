"""Independent branch and mixed-split contracts for EXP-FWD-022.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from pathlib import Path
import json
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest
import yaml

from phc2d.data.full_cell_mixed import file_sha256, materialize_mixed_validation
from phc2d.forward.global_local_branch_capability import finalize, freeze, parameter_controls, submit, task


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


def test_authorized_head_fallback_matches_parameters_without_changing_ordinary_heads():
    pytest.importorskip("torch")
    controller = yaml.safe_load(Path("configs/research/full_cell_global_local_bridge_exp_fwd_020_v1.yaml").read_text())
    params = {**controller["model"]["shared_params"], "local_kernel": 3}
    rows = parameter_controls(params, (64, 64), (81, 10), {"min": 8, "max": 512, "step": 8},
                             0.05, {"min": 8, "max": 2048, "step": 1})
    assert [r["parameter_count"] for r in rows] == [2571978, 2369610, 256170, 2572395, 2614890]
    assert [r["dense_head_hidden_dim"] for r in rows] == [None, None, None, 291, None]
    assert all(abs(r["parameter_count"] / rows[0]["parameter_count"] - 1) <= 0.05 for r in rows[3:])


def test_optional_dense_hidden_head_forward_backward_and_legacy_schema():
    torch = pytest.importorskip("torch")
    from phc2d.models import build_model

    kwargs = dict(input_shape=(4, 4), output_shape=(2, 3), width=8, depth=1,
                  modes=(2, 2), norm_groups=2, local_kernel=3)
    torch.manual_seed(9)
    legacy = build_model("global_local_band_operator_v1", **kwargs)
    torch.manual_seed(9)
    explicit = build_model("global_local_band_operator_v1", **kwargs, dense_head_hidden_dim=None)
    assert all(torch.equal(x, explicit.state_dict()[k]) for k, x in legacy.state_dict().items())
    hidden = build_model("global_local_band_operator_v1", **kwargs, dense_head_hidden_dim=13, enabled_branches=["spectral"])
    y = hidden(torch.rand(2, 1, 4, 4))
    assert y.shape == (2, 2, 3) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert hidden.head.projection[0].out_features == 13
    assert torch.isfinite(hidden.head.projection[0].weight.grad).all()
    with pytest.raises(ValueError, match="positive"):
        build_model("global_local_band_operator_v1", **kwargs, dense_head_hidden_dim=0)


def submission_campaign(tmp_path, status="capability_frozen_pending_preflight"):
    config = yaml.safe_load(Path("configs/research/full_cell_branch_capability_exp_fwd_022_v1.yaml").read_text())
    resolved = tmp_path / "config.resolved.yaml"
    resolved.write_text(yaml.safe_dump(config))
    (tmp_path / "freeze.json").write_text(json.dumps({"resolved_contract_sha256": file_sha256(resolved)}))
    (tmp_path / "campaign_status.json").write_text(json.dumps({"status": status}))
    return config["authorization_id"]


def test_preflight_submission_is_exclusive_and_uses_shared_runtime(tmp_path, monkeypatch):
    auth = submission_campaign(tmp_path)
    calls = []
    def scheduler(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="123;cluster\n")
    monkeypatch.setattr("subprocess.run", scheduler)
    result = submit(tmp_path, "preflight", tmp_path, auth)
    assert result["preflight_job_id"] == "123"
    assert "PHC2D_CUDA_MODULE=cuda/11.8" in " ".join(calls[0])
    assert calls[0][-1].endswith("jobs/paracloud_exp_fwd_022_capability.sh")
    with pytest.raises(FileExistsError):
        submit(tmp_path, "preflight", tmp_path, auth)
    assert len(calls) == 1


def test_first_training_batch_requires_accounting_and_records_only_15_tasks(tmp_path, monkeypatch):
    auth = submission_campaign(tmp_path, "capability_preflight_passed")
    (tmp_path / "submission_preflight.json").write_text(json.dumps({"preflight_job_id": "123", "status": "submitted"}))
    calls = []
    def scheduler(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="123|COMPLETED|0:0\n" if args[0] == "sacct" else str(200 + len(calls)))
    monkeypatch.setattr("subprocess.run", scheduler)
    result = submit(tmp_path, "training", tmp_path, auth)
    assert result["train_job_id"] == "202" and "finalize_job_id" not in result
    assert "--array=0-14%3" in calls[1] and len(calls) == 2
    assert json.loads((tmp_path / "campaign_status.json").read_text())["status"] == "capability_training_batch_0_submitted"


def test_second_batch_requires_all_previous_tasks_and_adds_finalizer(tmp_path, monkeypatch):
    auth = submission_campaign(tmp_path, "capability_training_batch_0_submitted")
    (tmp_path / "submission_training_batch_0.json").write_text(json.dumps({"train_job_id": "123", "status": "submitted"}))
    calls, audited = [], []
    def scheduler(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="\n".join(f"123_{i}|COMPLETED|0:0" for i in range(15))
                               if args[0] == "sacct" else str(200 + len(calls)))
    monkeypatch.setattr("subprocess.run", scheduler)
    monkeypatch.setattr("phc2d.forward.global_local_branch_capability.task", lambda campaign, i: {"task_index": i})
    monkeypatch.setattr("phc2d.forward.global_local_branch_capability.audit_training_task", lambda row: audited.append(row["task_index"]))
    result = submit(tmp_path, "training", tmp_path, auth, batch_index=1)
    assert audited == list(range(15))
    assert "--array=15-29%3" in calls[1] and "--dependency=afterok:202" in calls[2]
    assert result["finalize_job_id"] == "203"


def test_second_batch_does_not_start_after_failed_task(tmp_path, monkeypatch):
    auth = submission_campaign(tmp_path, "capability_training_batch_0_submitted")
    (tmp_path / "submission_training_batch_0.json").write_text(json.dumps({"train_job_id": "123", "status": "submitted"}))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: SimpleNamespace(stdout="123_0|FAILED|1:0"))
    with pytest.raises(RuntimeError, match="previous-stage accounting"):
        submit(tmp_path, "training", tmp_path, auth, batch_index=1)
    assert not (tmp_path / "submission_training_batch_1.json").exists()


def test_failed_submission_intent_is_preserved_and_never_automatically_retried(tmp_path, monkeypatch):
    auth = submission_campaign(tmp_path)
    def scheduler(*args, **kwargs):
        raise RuntimeError("uncertain scheduler response")
    monkeypatch.setattr("subprocess.run", scheduler)
    with pytest.raises(RuntimeError, match="uncertain"):
        submit(tmp_path, "preflight", tmp_path, auth)
    assert json.loads((tmp_path / "submission_preflight.json").read_text())["status"] == "submission_failed_or_uncertain"
    with pytest.raises(FileExistsError):
        submit(tmp_path, "preflight", tmp_path, auth)


def test_submission_rejects_unapproved_identity_and_hash_drift(tmp_path):
    auth = submission_campaign(tmp_path)
    with pytest.raises(RuntimeError, match="authorization"):
        submit(tmp_path, "preflight", tmp_path, "unapproved")
    with (tmp_path / "config.resolved.yaml").open("a") as h:
        h.write("\n# drift\n")
    with pytest.raises(RuntimeError, match="hash drift"):
        submit(tmp_path, "preflight", tmp_path, auth)


def test_platform_default_memory_omits_explicit_memory_request(tmp_path, monkeypatch):
    auth = submission_campaign(tmp_path)
    path = tmp_path / "config.resolved.yaml"
    config = yaml.safe_load(path.read_text())
    config["runtime"]["memory_mb"] = None
    path.write_text(yaml.safe_dump(config))
    (tmp_path / "freeze.json").write_text(json.dumps({"resolved_contract_sha256": file_sha256(path)}))
    calls = []
    def scheduler(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="123")
    monkeypatch.setattr("subprocess.run", scheduler)
    submit(tmp_path, "preflight", tmp_path, auth)
    assert not any(arg.startswith("--mem") for arg in calls[0])


def test_scheduler_failure_preserves_diagnostic_stderr(tmp_path, monkeypatch):
    auth = submission_campaign(tmp_path)
    def scheduler(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, output="", stderr="explicit memory request prohibited")
    monkeypatch.setattr("subprocess.run", scheduler)
    with pytest.raises(subprocess.CalledProcessError):
        submit(tmp_path, "preflight", tmp_path, auth)
    marker = json.loads((tmp_path / "submission_preflight.json").read_text())
    assert marker["scheduler_stderr"] == "explicit memory request prohibited"
    assert marker["status"] == "submission_failed_or_uncertain"


def test_worker_initializes_cuda_through_existing_runtime():
    worker = Path("jobs/paracloud_exp_fwd_022_capability.sh").read_text()
    assert 'source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"' in worker
    assert "SLURM_ARRAY_TASK_ID" in worker


def test_freeze_creates_unique_matrix_and_task_rejects_config_drift(tmp_path):
    pytest.importorskip("torch")
    params = dict(width=8, depth=1, modes=[2, 2], norm_groups=2, head_kind="dense")
    datasets = {name: source(tmp_path, name) for name in ("k5_binary", "k20_binary", "k5_graded", "k20_graded")}
    controller = tmp_path / "source.yaml"
    controller.write_text(yaml.safe_dump({"model": {"shared_params": params}, "datasets": datasets}))
    config = yaml.safe_load(Path("configs/research/full_cell_branch_capability_exp_fwd_022_v1.yaml").read_text())
    config.update(source_controller="source.yaml", width_search={"min": 8, "max": 64, "step": 2}, parameter_tolerance=0.99)
    config["split"].update(train_per_domain=4, validation_per_domain=1)
    contract = tmp_path / "contract.yaml"
    contract.write_text(yaml.safe_dump(config))
    campaign = freeze(contract, tmp_path / "campaign", tmp_path)
    snapshot = json.loads((campaign / "freeze.json").read_text())
    assert snapshot["task_count"] == 30 and snapshot["test_executed"] is False
    assert snapshot["resolved_contract_sha256"] == file_sha256(campaign / "config.resolved.yaml")
    rows = [task(campaign, i) for i in range(30)]
    assert len({r["case_id"] for r in rows}) == 30
    derived = yaml.safe_load(Path(rows[0]["config"]).read_text())
    assert derived["training"]["train_size"] == 8
    assert derived["models"]["global_local_band_operator_v1"]["dense_head_hidden_dim"] is None
    with Path(rows[0]["config"]).open("a") as h:
        h.write("\n# modified\n")
    with pytest.raises(RuntimeError, match="config hash drift"):
        task(campaign, 0)


def test_finalize_rejects_incomplete_matrix_before_evaluation(tmp_path):
    (tmp_path / "task_manifest.csv").write_text("task_index,case_id,lane,variant,seed\n")
    with pytest.raises(RuntimeError, match="complete unique 30-task"):
        finalize(tmp_path)
