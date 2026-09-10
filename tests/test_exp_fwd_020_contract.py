"""Contract tests for EXP-FWD-020 global-local bridge.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from pathlib import Path
import csv
import json

import pytest
import yaml

from phc2d.models import build_model, get_model_spec
from phc2d.forward import full_cell
from phc2d.forward.global_local_bridge_campaign import freeze_exp_fwd_020_continuations, freeze_g1


ROOT = Path(__file__).resolve().parents[1]


def test_exp_fwd_020_is_authorized_for_the_gated_n56_campaign() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/research/full_cell_global_local_bridge_exp_fwd_020_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["experiment"] == {
        "experiment_id": "EXP-FWD-020",
        "family": "full_cell_global_local_bridge",
        "evidence_status": "authorized_n56_gated_campaign_pending_preflight",
        "primary_layer": "L3_experiment_evidence",
    }
    assert config["split"]["test_targets_loaded"] is False
    assert config["model"]["paired_variants"] == {
        "baseline": {"local_kernel": 3},
        "candidate": {"local_kernel": 7},
    }
    assert config["authorization"]["granted"] is True
    assert config["runtime"]["target_platform"] == "paracloud-n56"
    assert config["runtime"]["partition"] == "gpu_5090"
    assert config["runtime"]["cuda_module"] == "cuda/12.8"
    assert config["runtime"]["torch_version"] == "2.7.1"
    assert config["runtime"]["environment"] == "/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1"
    assert config["runtime"]["max_concurrent_jobs"] == 8
    assert len(config["datasets"]) == 6
    assert all("n40_path" in item and "n56_path" in item for item in config["datasets"].values())

    spec = get_model_spec("global_local_band_operator_v1")
    assert spec.accepts_k_points is True
    assert spec.accepts_lane_id is False
    assert spec.defaults["local_kernel"] == 7


def test_exp_fwd_020_kernel_pair_stays_within_frozen_capacity_gate() -> None:
    pytest.importorskip("torch")
    baseline = build_model("global_local_band_operator_v1", local_kernel=3)
    candidate = build_model("global_local_band_operator_v1", local_kernel=7)
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
    assert baseline_count == 2_571_978
    assert candidate_count == 2_587_338
    assert candidate_count / baseline_count <= 1.05


def test_exp_fwd_020_g1_expands_exactly_six_domains_by_two_variants(tmp_path: Path) -> None:
    out = freeze_g1(ROOT / "configs/research/full_cell_global_local_bridge_exp_fwd_020_v1.yaml", tmp_path / "g1", repository_root=ROOT)
    with (out / "task_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert {row["domain"] for row in rows} == {"k5_binary", "k5_graded", "k20_binary", "k20_graded", "f50_binary", "f50_graded"}
    assert {row["variant"] for row in rows} == {"baseline", "candidate"}
    assert {row["seed"] for row in rows} == {"0"}
    resolved = yaml.safe_load((ROOT / rows[0]["config"]).read_text(encoding="utf-8"))
    assert resolved["experiment"]["held_out_test_authorized"] is False
    assert str(resolved["dataset"]["path"]).startswith("/data/run01/scxl892/data/exp_fwd_020/")


def test_exp_fwd_020_submitter_requires_and_propagates_versioned_environment() -> None:
    submitter = (ROOT / "jobs/paracloud_submit_exp_fwd_020.sh").read_text(encoding="utf-8")
    preflight = (ROOT / "jobs/paracloud_exp_fwd_020_preflight.sh").read_text(encoding="utf-8")
    assert ': "${PHC2D_ENV:?Set the versioned runtime environment directory.}"' in submitter
    assert '"$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_bridge_campaign' in submitter
    assert 'PHC2D_ENV=$PHC2D_ENV' in submitter
    assert "required PyTorch 2.7.1 evidence missing" in submitter
    assert "-p gpu_5090 --gres=gpu:1 -c 8" in submitter
    assert "gpu_4090" not in submitter
    assert "PHC2D_CUDA_MODULE=cuda/12.8" in preflight
    assert 'torch.__version__.startswith("2.7.1")' in preflight
    assert "freeze-continuations" in submitter
    assert "EXP-FWD-020_epoch300_continuations" in submitter


def test_exp_fwd_020_fixed_epoch300_continuation_is_validation_only(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/research/full_cell_global_local_bridge_exp_fwd_020_v1.yaml").read_text(encoding="utf-8")
    )
    derived = {
        "experiment": {"experiment_id": "EXP-FWD-020", "lane": "binary", "held_out_test_authorized": False},
        "dataset": {"path": "unused.h5", "grid_field": "material_area_fraction_grid_64"},
        "representation": {"sampling": "continuous", "threshold": None, "resolution": 64},
        "models": {"global_local_band_operator_v1": {**config["model"]["shared_params"], "local_kernel": 3}},
        "training": {**config["training"], "train_size": 8000, "max_epochs": 300, "required_stop_epoch": 300, "convergence_window": [281, 300], "convergence_metric": "validation_bands_2_10_mre", "convergence_relative_threshold": 0.005, "require_terminal_checkpoint": True, "early_stopping_patience": 0, "learning_rates": {"global_local_band_operator_v1": config["training"]["learning_rate"]}},
        "runtime": {"precision": "fp32", "amp_enabled": False, "tf32_enabled": False},
        "continuation": {"mode": "epoch200_weights_optimizer_reset", "source_experiment": "EXP-FWD-020", "source_epoch": 200, "optimizer_reset": True, "data_order_state_restored": False, "check_interval": 20, "relative_threshold": 0.005, "max_total_epoch": 300, "stop_when_converged": False, "source_run": str(tmp_path / "source")},
    }
    resolved = full_cell._validate_and_resolve(derived, model_name="global_local_band_operator_v1", seed=0, admission_probe=False)
    assert resolved["training"]["max_epochs"] == 300
    assert resolved["experiment"]["held_out_test_authorized"] is False


def test_exp_fwd_020_freezes_exactly_29_epoch300_continuations(tmp_path: Path) -> None:
    source = tmp_path / "g2"; source.mkdir()
    rows = []
    domains = ("k5_binary", "k5_graded", "k20_binary", "k20_graded", "f50_binary", "f50_graded")
    for index, (domain, variant, seed) in enumerate((d, v, s) for d in domains for v in ("baseline", "candidate") for s in (0, 1, 2)):
        run = source / "runs" / f"{domain}_{variant}_s{seed}"; run.mkdir(parents=True)
        model, terminal = run / "model.pt", run / "epoch200.pt"
        model.write_bytes(b"best" + bytes([index])); terminal.write_bytes(b"terminal" + bytes([index]))
        for name in ("run.log", "run.meta.json", "metrics.json"):
            (run / name).write_text("{}\n", encoding="utf-8")
        (run / "config.resolved.yaml").write_text("experiment: {}\n", encoding="utf-8")
        (run / "history.csv").write_text("epoch\n" + "\n".join(str(epoch) for epoch in range(1, 201)) + "\n", encoding="utf-8")
        result = {"status": "formal_training_success" if index < 7 else "training_complete_convergence_not_met", "test_executed": False, "selection_split": "validation", "stop_epoch": 200, "history_epoch_count": 200, "checkpoint_sha256": __import__("hashlib").sha256(model.read_bytes()).hexdigest(), "terminal_checkpoint_sha256": __import__("hashlib").sha256(terminal.read_bytes()).hexdigest()}
        (run / "results.json").write_text(json.dumps(result), encoding="utf-8")
        config = source / "configs" / f"{index}.yaml"; config.parent.mkdir(exist_ok=True)
        config.write_text(yaml.safe_dump({"experiment": {"experiment_id": "EXP-FWD-020", "held_out_test_authorized": False}, "training": {"max_epochs": 200}}), encoding="utf-8")
        rows.append({"task_index": index, "case_id": f"{domain}_{variant}_s{seed}", "domain": domain, "lane": "binary" if domain.endswith("binary") else "graded", "variant": variant, "seed": seed, "config": str(config), "run_dir": str(run)})
    with (source / "task_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (source / "candidate_gate.json").write_text(json.dumps({"selected_domains": list(domains)}), encoding="utf-8")
    frozen = freeze_exp_fwd_020_continuations(source, tmp_path / "continuations", repository_root=ROOT)
    with (frozen / "task_manifest.csv").open(encoding="utf-8", newline="") as handle:
        continuation_rows = list(csv.DictReader(handle))
    assert len(continuation_rows) == 29
    continuation = yaml.safe_load(Path(continuation_rows[0]["config"]).read_text(encoding="utf-8"))
    assert continuation["training"]["required_stop_epoch"] == 300
    assert continuation["continuation"]["stop_when_converged"] is False


def test_exp_fwd_020_g3_contract_is_exact_exploratory_216_cell_matrix() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/research/full_cell_global_local_bridge_exp_fwd_020_g3_v1.yaml").read_text(encoding="utf-8")
    )
    assert config["authorization"]["held_out_test_authorized"] is True
    assert config["experiment"]["promotion_eligible"] is False
    assert config["matrix"] == {
        "checkpoint_count": 36,
        "target_domain_count": 6,
        "cell_count": 216,
        "relationship_counts": {"native": 36, "cross_lane": 36, "cross_dataset": 72, "cross_dataset_cross_lane": 72},
        "bootstrap_replicates": 10000,
    }
