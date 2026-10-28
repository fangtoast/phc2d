"""Frozen partition, metric and admission contracts for the official comparison.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import json
from pathlib import Path
import pytest
from phc2d.forward.official_comparison_campaign import (
    task_configs, load_profile, claim_task, audit_global_release, audit_run,
)


def profile(platform):
    return load_profile(Path(f"configs/research/official_comparison_exp_fwd_031_{platform}_v1.yaml"))


def test_partition_and_explicit_metrics():
    all_cases = []
    for platform in ("n40", "n56"):
        tasks = task_configs(profile(platform))
        assert len(tasks) == 18
        assert sum(seed == 0 for _, _, seed, _ in tasks) == 6
        for case, model, seed, cfg in tasks:
            all_cases.append(case)
            assert not cfg.get("continuation")
            assert cfg["training"]["loss"] == "normalized_full_matrix_mse"
            assert cfg["training"]["checkpoint_selection"] == "validation_bands_2_10_mre"
            assert cfg["training"]["scheduler_metric"] == "val_bands_2_10_mre_pct"
            assert cfg["experiment"]["primary_reporting_metric"] == "bands_1_10_mre_pct"
    assert len(set(all_cases)) == 36


def test_config_drift_and_retry_rejected(tmp_path):
    cfg = profile("n40")
    cfg["training"]["checkpoint_selection"] = "validation_bands_1_10_mre"
    with pytest.raises(ValueError, match="historical"):
        task_configs(cfg)
    first = claim_task(tmp_path, "case", {})
    assert first is not None
    (first / "failure.json").write_text('{}')
    assert claim_task(tmp_path, "case", {}) is None
    with pytest.raises(FileNotFoundError):
        audit_global_release(tmp_path)


def test_admission_rejects_a_probe_without_optimizer_update(tmp_path):
    torch = pytest.importorskip("torch")
    (tmp_path / "config.resolved.yaml").write_text("model: {params: {}}\n")
    (tmp_path / "metrics.json").write_text('{"optimizer_steps": 0}')
    torch.save({}, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="optimizer update"):
        audit_run(tmp_path, {"model": "official_fno_band_v1"}, admission=True)


def test_scheduler_uses_bands2_and_reset_loads_selected_weights(tmp_path):
    torch = pytest.importorskip("torch")
    from phc2d.forward.adaptive_convergence import AdaptiveOptimization
    from phc2d.forward.adaptive_training_evidence import initialize_adaptive_phase
    protocol = profile("n40")["protocol"]
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=.0005)
    selected = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(dict(state_dict=selected, epoch=271), tmp_path / "model.pt")
    with torch.no_grad():
        model.weight.add_(1)
    initialize_adaptive_phase(model, opt, torch.Generator(), tmp_path, 0, torch)
    torch.testing.assert_close(model.weight, selected["weight"])
    assert json.loads((tmp_path / "adaptive_phase_start.json").read_text())["selected_weights_epoch"] == 271
    control = AdaptiveOptimization(opt, protocol, .0005, torch)
    control.before_epoch(301)
    seen = []
    control.scheduler.step = lambda value: seen.append(value)
    control.after_epoch([dict(epoch=301, val_bands_1_10_mre_pct=.1, val_bands_2_10_mre_pct=.7)])
    assert seen == [.7]
