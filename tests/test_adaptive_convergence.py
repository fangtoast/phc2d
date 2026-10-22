"""Adaptive convergence, warm-start compatibility and cache evidence contracts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.forward.adaptive_convergence import AdaptiveOptimization, convergence_record
from phc2d.forward.convergence_campaign import adaptive_config
from phc2d.forward.full_cell import _FullCellH5Dataset, run_full_cell_forward
from phc2d.forward.full_cell_training_contracts import resolve_full_cell_training_config


PROTOCOL = yaml.safe_load(Path("configs/research/full_cell_convergence_n56_v1.yaml").read_text())["protocol"]


def history(end=500, value=lambda epoch: 1.0):
    return [dict(epoch=e, val_bands_2_10_mre_pct=value(e), val_bands_1_10_mre_pct=value(e)) for e in range(1, end + 1)]


def test_convergence_requires_floor_duration_two_checks_and_both_curves():
    first = convergence_record(history(), PROTOCOL, floor_since=400, streak=0)
    assert not first["passed"] and first["streak"] == 1
    assert convergence_record(history(520), PROTOCOL, floor_since=400, streak=1)["passed"]
    assert not convergence_record(history(), PROTOCOL, floor_since=480, streak=1)["passed"]
    improving = history(value=lambda e: 2 - e * .001)
    assert not convergence_record(improving, PROTOCOL, floor_since=400, streak=1)["passed"]
    overfit = history(value=lambda e: .5 if e <= 300 else 1.0)
    assert not convergence_record(overfit, PROTOCOL, floor_since=400, streak=1)["passed"]
    primary_worse = history()
    for row in primary_worse[-20:]:
        row["val_bands_1_10_mre_pct"] = 1.1
    assert not convergence_record(primary_worse, PROTOCOL, floor_since=400, streak=1)["passed"]
    primary_worse[-1]["val_bands_1_10_mre_pct"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        convergence_record(primary_worse, PROTOCOL, floor_since=400, streak=1)


def test_scheduler_preserves_prefix_then_reduces_and_saves_full_state(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    controller = AdaptiveOptimization(opt, PROTOCOL, .001, torch)
    controller.before_epoch(300)
    assert opt.param_groups[0]["lr"] == .001 and controller.scheduler is None
    controller.before_epoch(301)
    assert opt.param_groups[0]["lr"] == pytest.approx(.0002)
    rows = history(300)
    for e in range(301, 601):
        rows.append(dict(epoch=e, val_bands_2_10_mre_pct=1., val_bands_1_10_mre_pct=1.))
        controller.after_epoch(rows)
    assert opt.param_groups[0]["lr"] == pytest.approx(.00001)
    controller.save(tmp_path / "state.pt", model, torch.Generator().manual_seed(0), rows, {})
    payload = torch.load(tmp_path / "state.pt", weights_only=True)
    assert payload["epoch"] == 600
    for key in ("optimizer_state_dict", "scheduler_state_dict", "python_rng_state", "numpy_rng_state", "torch_rng_state", "data_order_generator_state", "controller_state"):
        assert key in payload


def tiny_config(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "data.h5"
    rng = np.random.default_rng(3)
    with h5py.File(path, "w") as h:
        h["grid"] = rng.random((10, 4, 4), dtype=np.float32)
        labels = .2 + rng.random((10, 3, 2), dtype=np.float32)
        labels[-2:] = np.nan
        h["Wh"] = labels
        h["idx_train"] = np.arange(6)
        h["idx_val"] = np.arange(6, 8)
        h["idx_test"] = np.arange(8, 10)
        h["k_points"] = rng.random((3, 2), dtype=np.float32)
    return dict(experiment=dict(experiment_id="EXP-FWD-028", lane="graded"),
                dataset=dict(path=str(path), grid_field="grid"),
                representation=dict(resolution=4, sampling="continuous", threshold=None, physical_equivalence=True),
                models=dict(mlp=dict(hidden_dim=8, depth=1)),
                training=dict(train_size=6, max_epochs=1000, optimizer="adamw", learning_rates=dict(mlp=.0003), weight_decay=.00001,
                              effective_batch_size=64, micro_batch_size=64, early_stopping_patience=0, epoch_100_continue_gate=False),
                adaptive_optimization=copy.deepcopy(PROTOCOL),
                runtime=dict(device="cpu", require_accelerator=False, precision="fp32", amp_enabled=False, tf32_enabled=False, num_workers=0, cache_in_memory=True), output={})


def test_cpu_admission_and_epoch300_best_warm_start_without_test_reads(tmp_path):
    torch = pytest.importorskip("torch")
    cfg = tiny_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    source = tmp_path / "source"
    run_full_cell_forward(config_path, model_name="mlp", seed=0, run_dir=source, admission_probe=True)
    result = json.loads((source / "results.json").read_text())
    assert result["status"] == "full_cell_forward_admission_probe_pass" and not result["test_executed"]
    resolved = yaml.safe_load((source / "config.resolved.yaml").read_text())
    resolved["experiment"]["experiment_id"] = "EXP-FWD-024_n6_e300"
    (source / "config.resolved.yaml").write_text(yaml.safe_dump(resolved))
    with (source / "history.csv").open() as handle:
        original = list(csv.DictReader(handle))[0]
    with (source / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=original.keys()); writer.writeheader()
        writer.writerows(dict(original, epoch=e) for e in range(1, 301))
    terminal = torch.load(source / "model.pt", weights_only=True)
    terminal["epoch"] = 300
    torch.save(terminal, source / "epoch300.pt")
    import hashlib
    result.update(stop_epoch=300, history_epoch_count=300, terminal_checkpoint_sha256=hashlib.sha256((source / "epoch300.pt").read_bytes()).hexdigest())
    (source / "results.json").write_text(json.dumps(result))
    extended = adaptive_config(cfg, "EXP-FWD-026", PROTOCOL, source, "EXP-FWD-024")
    config_path.write_text(yaml.safe_dump(extended))
    run_full_cell_forward(config_path, model_name="mlp", seed=0, run_dir=tmp_path / "continued", admission_probe=True)
    report = json.loads((tmp_path / "continued/results.json").read_text())
    assert report["stop_epoch"] == 301 and report["warm_start"]["initial_weights_epoch"] == 1
    assert report["warm_start"]["source_scientific_contract_verified"]
    assert not report["warm_start"]["optimizer_state_restored"]


def test_in_memory_cache_matches_lazy_training_inputs(tmp_path):
    cfg = tiny_config(tmp_path)
    args = dict(path=cfg["dataset"]["path"], indices=np.array([4, 1, 3, 0, 5, 2]), grid_field="grid", layout="flat",
                sampling="continuous", threshold=None, target_mean=np.zeros((3, 2), dtype=np.float32), target_std=np.ones((3, 2), dtype=np.float32))
    lazy, cache = _FullCellH5Dataset(**args), _FullCellH5Dataset(**args, cache_in_memory=True)
    for i in range(6):
        for left, right in zip(lazy[i], cache[i]):
            np.testing.assert_array_equal(left, right)


def test_adaptive_contract_does_not_weaken_fixed_budget(tmp_path):
    cfg = tiny_config(tmp_path)
    resolve_full_cell_training_config(cfg, model_name="mlp", seed=2, admission_probe=False)
    cfg["experiment"]["experiment_id"] = "EXP-FWD-024"
    with pytest.raises(ValueError, match="adaptive optimization"):
        resolve_full_cell_training_config(cfg, model_name="mlp", seed=2, admission_probe=False)


def test_cpu_training_writes_adaptive_terminal_and_separate_selected_weights(tmp_path):
    pytest.importorskip("torch")
    cfg = tiny_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "training"
    run_full_cell_forward(config_path, model_name="mlp", seed=0, run_dir=out)
    report = json.loads((out / "results.json").read_text())
    assert 400 <= report["stop_epoch"] <= 1000
    assert Path(report["terminal_checkpoint_path"]).is_file()
    assert (out / "training_state.pt").is_file()
    assert report["best_epoch"] <= report["stop_epoch"] and not report["test_executed"]
    if report["stop_epoch"] < 1000:
        assert report["convergence"]["passed"]


def test_validation_summary_uses_sample_sd_and_rejects_incomplete_seeds(tmp_path, monkeypatch):
    from phc2d.forward import convergence_results as results
    from phc2d.forward.budget_confirmation import file_digest
    rows = []
    for seed, primary in enumerate([1., 2., 3.]):
        run = tmp_path / str(seed)
        run.mkdir()
        for name in ("model.pt", "epoch1000.pt", "training_state.pt", "config.resolved.yaml", "run.meta.json"):
            (run / name).write_text("fixture")
        (run / "history.csv").write_text("epoch,train_loss,val_bands_1_10_mre_pct,val_bands_2_10_mre_pct\n" +
                                         "".join(f"{e},1,1,1\n" for e in range(1, 1001)))
        (run / "results.json").write_text(json.dumps(dict(stop_epoch=1000, model="mlp", seed=seed,
            test_executed=False, selection_split="validation", checkpoint_sha256=file_digest(run / "model.pt"),
            terminal_checkpoint_sha256=file_digest(run / "epoch1000.pt"), terminal_checkpoint_path=str(run / "epoch1000.pt"),
            best_epoch=1, convergence=dict(passed=False))))
        (run / "metrics.json").write_text(json.dumps(dict(bands_1_10_mre_pct=primary)))
        rows.append(dict(case_id=f"domain_mlp_s{seed}", model="mlp", seed=seed, run_dir=str(run)))
    monkeypatch.setattr(results, "read_tasks", lambda campaign: rows)
    assert results.summarize_campaign(tmp_path)["converged_count"] == 0
    with (tmp_path / "validation_primary_summary.csv").open() as handle:
        summary = next(csv.DictReader(handle))
    assert float(summary["bands_1_10_mre_pct_mean"]) == 2.
    assert float(summary["bands_1_10_mre_pct_seed_sd"]) == 1.
    rows.pop()
    with pytest.raises(ValueError, match="three-seed"):
        results.summarize_campaign(tmp_path)


def test_scheduler_dependency_and_duplicate_submission_guard(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("submit_convergence", "jobs/paracloud_submit_convergence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = yaml.safe_load(Path("configs/research/full_cell_convergence_n56_v1.yaml").read_text())
    (tmp_path / "config.resolved.yaml").write_text(yaml.safe_dump(profile))
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    module.submit(tmp_path, tmp_path, "train", "12344")
    assert "--array=0-7%8" in calls[0] and "--dependency=afterok:12344" in calls[0]
    assert "--no-requeue" in calls[0] and "7-00:00:00" in calls[0]
    with pytest.raises(FileExistsError):
        module.submit(tmp_path, tmp_path, "train", "12344")
    assert len(calls) == 1
