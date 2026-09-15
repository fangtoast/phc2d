"""Contracts and smoke coverage for EXP-FWD-021 multi-domain training.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.forward import full_cell_multidomain_contracts as contracts
from phc2d.forward.full_cell_multidomain_evaluation import (
    evaluation_plan_rows,
    run_evaluation_task,
)
from phc2d.forward.full_cell_multidomain_training import forward_full_wh, run_training_task
from phc2d.models import build_model, get_model_spec


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/full_cell_multidomain_exp_fwd_021_v1.yaml"


def _profile() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_frozen_profile_has_exact_7_by_9_by_6_contract() -> None:
    profile = _profile()
    contracts._validate_campaign_shape(profile)
    assert len(profile["regimes"]) == 7
    assert len(profile["models"]) == 9
    assert len(profile["evaluation"]["targets"]) == 6
    assert sum(len(profile["models"]) for _ in profile["regimes"]) == 63
    assert profile["models"]["global_local_band_operator_v1"]["local_kernel"] == 3
    assert profile["training"]["seed"] == 0
    assert profile["training"]["max_epochs"] == 200
    assert profile["runtime"]["memory_policy"] == "scheduler_default_per_gpu"
    assert profile["runtime"]["aggregate_gpu_limit"] == 7
    assert profile["runtime"]["single_task_distributed_training"] is False
    for regime in profile["regimes"].values():
        assert sum(regime["domains"].values()) == 8000


def test_freeze_generates_nested_63_task_manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _profile()
    config = tmp_path / "profile.yaml"
    config.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    def fake_audit(domain, contract, path, verify_file_hash=True):
        del verify_file_hash
        return {
            "domain": domain,
            "label": contract["label"],
            "dataset_path": str(path),
            "grid_field": contract["grid_field"],
            "dataset_sha256": contract["sha256"],
            "sample_count": contract["sample_count"],
            "train_count": contract["train_count"],
            "validation_count": contract["validation_count"],
            "test_count": contract["test_count"],
            "target_shape": "[81, 10]",
            "grid_shape": "[64, 64]",
            "idx_train_hash": f"train-{domain}",
            "idx_val_hash": f"val-{domain}",
            "idx_test_hash": f"test-{domain}",
            "k_points_sha256": "shared-k-points",
            "test_targets_loaded": False,
        }

    monkeypatch.setattr(contracts, "audit_dataset", fake_audit)
    monkeypatch.setattr(
        contracts,
        "_selected_indices",
        lambda dataset_path, *, count, salt: np.arange(count, dtype=np.int64),
    )
    monkeypatch.setattr(
        contracts,
        "_pooled_normalizer",
        lambda datasets, selections: (
            np.zeros((81, 10), dtype=np.float32),
            np.ones((81, 10), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        contracts,
        "_subset_provenance",
        lambda dataset_path, indices: {
            "selected_sample_ids_sha256": f"selected-{len(indices)}",
            "validation_overlap_count": 0,
            "test_overlap_count": 0,
        },
    )
    campaign = contracts.freeze_campaign(
        config,
        tmp_path / "campaign",
        repository_root=ROOT,
        dataset_path_key="n56_path",
    )
    tasks = contracts.read_csv(campaign / "task_manifest.csv")
    subsets = contracts.read_csv(campaign / "subset_manifest.csv")
    assert len(tasks) == 63
    assert len({row["task_id"] for row in tasks}) == 63
    assert all(int(row["training_sample_count"]) == 8000 for row in tasks)
    assert all(int(row["validation_overlap_count"]) == 0 for row in subsets)
    assert all(int(row["test_overlap_count"]) == 0 for row in subsets)
    for domain in profile["datasets"]:
        arrays = [
            np.load(campaign / row["indices_path"])
            for row in subsets
            if row["domain"] == domain
        ]
        largest = max(arrays, key=len)
        assert all(set(array).issubset(set(largest)) for array in arrays)


def test_pooled_normalizer_reads_only_selected_training_samples(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    datasets = {}
    selections = {}
    for domain, offset in (("a", 1.0), ("b", 3.0)):
        path = tmp_path / f"{domain}.h5"
        with h5py.File(path, "w") as handle:
            values = np.full((4, 2, 3), 1.0e6, dtype=np.float32)
            values[0] = offset
            values[1] = offset + 1.0
            handle.create_dataset("Wh", data=values)
        datasets[domain] = {"dataset_path": str(path)}
        selections[domain] = np.asarray([0, 1], dtype=np.int64)
    mean, std = contracts._pooled_normalizer(datasets, selections)
    assert np.allclose(mean, 2.5)
    assert np.allclose(std, np.std([1.0, 2.0, 3.0, 4.0]))


def test_audit_dataset_checks_shapes_splits_and_does_not_load_test_targets(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("grid", data=np.zeros((6, 4, 5), dtype=np.float32))
        handle.create_dataset("Wh", data=np.ones((6, 3, 10), dtype=np.float32))
        handle.create_dataset("k_points", data=np.zeros((3, 2), dtype=np.float32))
        handle.create_dataset("idx_train", data=np.asarray([0, 1, 2]))
        handle.create_dataset("idx_val", data=np.asarray([3]))
        handle.create_dataset("idx_test", data=np.asarray([4, 5]))
        handle.create_dataset("sample_id", data=np.asarray([f"s{i}".encode() for i in range(6)]))
    contract = {
        "sha256": contracts.sha256_file(path),
        "grid_field": "grid",
        "sample_count": 6,
        "train_count": 3,
        "validation_count": 1,
        "test_count": 2,
        "target_shape": [3, 10],
        "grid_shape": [4, 5],
    }
    result = contracts.audit_dataset("tiny", contract, path)
    assert result["test_targets_loaded"] is False
    assert result["target_shape"] == "[3, 10]"


def test_evaluation_plan_is_exactly_378_unique_cells() -> None:
    training = [
        {
            "task_id": f"task-{index}",
            "regime": f"regime-{index // 9}",
            "model": f"model-{index % 9}",
            "seed": 0,
            "checkpoint_path": f"/checkpoint/{index}.pt",
            "checkpoint_sha256": f"hash-{index}",
        }
        for index in range(63)
    ]
    rows = evaluation_plan_rows(training, [f"target-{index}" for index in range(6)])
    assert len(rows) == 378
    assert len({row["cell_id"] for row in rows}) == 378
    assert {int(row["cell_index"]) for row in rows} == set(range(378))


def test_n56_scripts_freeze_resource_and_no_retry_contract() -> None:
    train = (ROOT / "jobs/paracloud_exp_fwd_021_train.sh").read_text(encoding="utf-8")
    evaluate = (ROOT / "jobs/paracloud_exp_fwd_021_evaluate.sh").read_text(encoding="utf-8")
    controller = (ROOT / "jobs/paracloud_submit_exp_fwd_021.sh").read_text(encoding="utf-8")
    for worker in (train, evaluate):
        assert "#SBATCH --partition=gpu_5090" in worker
        assert "#SBATCH --gres=gpu:1" in worker
        assert "#SBATCH --cpus-per-task=8" in worker
        assert "#SBATCH --mem" not in worker
        assert "#SBATCH --time=24:00:00" in worker
        assert "cuda/12.8" in worker
        assert "phc2d_gpu_cu128_torch271_r1" in worker
    assert "%7" in controller
    assert "start + 15" in controller
    assert "no retry permitted" in controller
    assert "gpu_5090" not in controller
    assert "N40" not in controller


@pytest.mark.parametrize("model_name", list(_profile()["models"]))
def test_all_nine_models_complete_cpu_small_forward(model_name: str) -> None:
    torch = pytest.importorskip("torch")
    profile = _profile()
    spec = get_model_spec(model_name)
    params = dict(profile["models"][model_name])
    kwargs = {**params, "input_shape": (64, 64), "output_shape": (81, 10)}
    k_points = torch.zeros((81, 2), dtype=torch.float32)
    if spec.accepts_k_points:
        kwargs["k_points"] = k_points.numpy()
    model = build_model(model_name, **kwargs)
    inputs = torch.zeros((1, 4096), dtype=torch.float32) if spec.input_layout == "flat" else torch.zeros((1, 64, 64), dtype=torch.float32)
    with torch.no_grad():
        output = forward_full_wh(
            model,
            inputs,
            model_spec=spec,
            k_points=k_points,
            lane_index=None,
            torch=torch,
        )
    assert tuple(output.shape) == (1, 81, 10)
    assert torch.isfinite(output).all()


def test_cpu_admission_writes_complete_artifacts_without_test_access(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("torch")
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    datasets = []
    subset_rows = []
    selected = np.arange(4000, dtype=np.int64)
    target_values = []
    k_points = np.zeros((3, 2), dtype=np.float32)
    for domain, offset in (("a_binary", 1.0), ("b_binary", 2.0)):
        path = tmp_path / f"{domain}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("grid", data=np.zeros((4003, 4, 4), dtype=np.float32))
            targets = np.full((4003, 3, 10), offset, dtype=np.float32)
            handle.create_dataset("Wh", data=targets)
            handle.create_dataset("k_points", data=k_points)
            handle.create_dataset("idx_train", data=selected)
            handle.create_dataset("idx_val", data=np.asarray([4000, 4001]))
            handle.create_dataset("idx_test", data=np.asarray([4002]))
            handle.create_dataset("sample_id", data=np.asarray([f"{domain}-{i}".encode() for i in range(4003)]))
        dataset_hash = contracts.sha256_file(path)
        datasets.append(
            {
                "domain": domain,
                "dataset_path": str(path),
                "grid_field": "grid",
                "dataset_sha256": dataset_hash,
                "k_points_sha256": contracts.sha256_array(k_points, dtype=np.float32),
            }
        )
        relative = Path("subsets") / "SMOKE" / f"{domain}.npy"
        (campaign / relative).parent.mkdir(parents=True, exist_ok=True)
        np.save(campaign / relative, selected)
        subset_rows.append(
            {
                "regime": "SMOKE",
                "domain": domain,
                "sample_count": 4000,
                "indices_path": relative.as_posix(),
                "indices_sha256": contracts.sha256_array(selected, dtype=np.int64),
            }
        )
        target_values.append(np.full((4000, 3, 10), offset, dtype=np.float32))
    pooled = np.concatenate(target_values)
    mean = pooled.mean(axis=0).astype(np.float32)
    std = np.maximum(pooled.std(axis=0).astype(np.float32), 1.0e-6)
    normalizer = campaign / "normalizers/SMOKE.npz"
    normalizer.parent.mkdir()
    np.savez_compressed(normalizer, target_mean=mean, target_std=std)
    contracts.write_csv(campaign / "dataset_manifest.csv", datasets)
    contracts.write_csv(campaign / "subset_manifest.csv", subset_rows)
    contracts.write_csv(
        campaign / "normalizer_manifest.csv",
        [
            {
                "regime": "SMOKE",
                "normalizer_path": "normalizers/SMOKE.npz",
                "normalizer_file_sha256": contracts.sha256_file(normalizer),
                "target_mean_sha256": contracts.sha256_array(mean, dtype=np.float32),
                "target_std_sha256": contracts.sha256_array(std, dtype=np.float32),
            }
        ],
    )
    contracts.write_csv(
        campaign / "task_manifest.csv",
        [
            {
                "task_index": 0,
                "task_id": "smoke_mlp_s0",
                "regime": "SMOKE",
                "model": "mlp",
                "seed": 0,
                "learning_rate": 0.001,
                "run_dir": "training/SMOKE/mlp_s0",
                "execution_start_git_commit": "test",
            }
        ],
    )
    profile = {
        "experiment": {"experiment_id": "EXP-FWD-021"},
        "regimes": {"SMOKE": {"domains": {"a_binary": 4000, "b_binary": 4000}}},
        "models": {"mlp": {"hidden_dim": 16, "depth": 2, "dropout": 0.0, "activation": "gelu"}},
        "training": {
            "max_epochs": 200,
            "weight_decay": 1.0e-5,
            "micro_batch_size": 8,
            "effective_batch_size": 8,
            "convergence_relative_threshold": 0.005,
        },
        "runtime": {
            "device": "cpu",
            "require_accelerator": False,
            "precision": "fp32",
            "amp_enabled": False,
            "tf32_enabled": False,
            "num_workers": 0,
            "pin_memory": False,
        },
    }
    (campaign / "config.resolved.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    output = run_training_task(
        campaign,
        0,
        admission_probe=True,
        device_override="cpu",
        max_train_batches=1,
        max_validation_samples=1,
    )
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert result["status"] == "multidomain_admission_probe_pass"
    assert result["test_executed"] is False
    for name in (
        "run.log",
        "run.meta.json",
        "config.resolved.yaml",
        "history.csv",
        "metrics.json",
        "results.json",
        "model.pt",
        "admission_epoch1.pt",
        "model_summary.json",
    ):
        assert (output / name).stat().st_size > 0

    evaluation = campaign / "evaluation"
    cache = evaluation / "cache/a_binary.npz"
    cache.parent.mkdir(parents=True)
    np.savez_compressed(
        cache,
        inputs=np.zeros((1, 4, 4), dtype=np.float32),
        targets=np.ones((1, 3, 10), dtype=np.float32),
        sample_ids=np.asarray(["a_binary-4002"]),
        indices=np.asarray([4002], dtype=np.int64),
        k_points=k_points,
    )
    contracts.write_csv(
        evaluation / "cache_manifest.csv",
        [
            {
                "target_domain": "a_binary",
                "cache_path": "cache/a_binary.npz",
                "cache_sha256": contracts.sha256_file(cache),
            }
        ],
    )
    contracts.write_csv(
        evaluation / "evaluation_manifest.csv",
        [
            {
                "cell_index": 0,
                "cell_id": "smoke_mlp_s0__to__a_binary",
                "task_id": "smoke_mlp_s0",
                "regime": "SMOKE",
                "model": "mlp",
                "seed": 0,
                "target_domain": "a_binary",
                "checkpoint_path": str(output / "model.pt"),
                "checkpoint_sha256": contracts.sha256_file(output / "model.pt"),
                "cell_dir": "evaluation/cells/SMOKE/mlp_s0/a_binary",
            }
        ],
    )
    cell = run_evaluation_task(campaign, 0, device_override="cpu")
    cell_result = json.loads((cell / "results.json").read_text(encoding="utf-8"))
    assert cell_result["status"] == "evaluation_cell_success"
    assert cell_result["per_band_record_count"] == 10
    assert len(contracts.read_csv(cell / "per_band_metrics.csv")) == 10
