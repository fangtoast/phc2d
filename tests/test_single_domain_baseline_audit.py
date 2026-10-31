"""Historical Full baseline provenance checks.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import hashlib
import json

import pytest
import yaml

from phc2d.forward.budget_confirmation import file_digest
from phc2d.forward.single_domain_baseline_audit import audit


def test_audit_checks_phase_provenance_and_rejects_changed_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import phc2d.models
    from phc2d.forward import single_domain_baseline_audit as module
    monkeypatch.setattr(phc2d.models, "build_model", lambda *a, **k: torch.nn.Linear(2, 1))
    mean, std = torch.zeros((2, 1)), torch.ones((2, 1))
    normalizer = hashlib.sha256(mean.numpy().tobytes() + std.numpy().tobytes()).hexdigest()
    dataset = dict(sha256="data", grid_field="grid", train_subset_hash="subset", normalizer_hash=normalizer,
                   k_points_hash="khash")
    monkeypatch.setattr(module, "inspect_dataset", lambda *a, **k: dataset)
    source = {"datasets": {"domain": {"sha256": "data", "grid_field": "grid"}},
              "models": {"global_local_band_operator_v1": {"width": 64, "local_kernel": 3}}}
    protocol = {"start_epoch": 300}
    manifest = dict(domains=["domain"], dataset_paths={"domain": "data.h5"}, candidates=[])
    training = dict(train_size=8000, train_subset_salt="exp-fwd-021-domain-ranking-v1", micro_batch_size=64,
                    effective_batch_size=64, optimizer="adamw", learning_rate=0.0005, weight_decay=0.00001,
                    loss="normalized_full_matrix_mse", checkpoint_selection="validation_bands_2_10_mre")
    for kernel in (3, 7):
        for seed in (0, 1, 2):
            run = tmp_path / f"{kernel}_{seed}"
            run.mkdir()
            config = dict(training=training, adaptive_optimization=protocol,
                          dataset=dict(expected_dataset_sha256="data", grid_field="grid"),
                          model=dict(params=dict(width=64, local_kernel=kernel)))
            (run / "config.resolved.yaml").write_text(yaml.safe_dump(config))
            (run / "history.csv").write_text("epoch,train_loss,val_bands_1_10_mre_pct,val_bands_2_10_mre_pct\n" +
                                             "".join(f"{i},1,1,1\n" for i in range(1, 401)))
            payload = dict(target_mean=mean, target_std=std, k_points_sha256="khash", input_shape=(2,),
                           output_shape=(1,), model_params={}, state_dict=torch.nn.Linear(2, 1).state_dict())
            torch.save(payload, run / "model.pt")
            result = dict(model="global_local_band_operator_v1", seed=seed, selection_split="validation",
                          checkpoint_role="best_validation", test_executed=False, stop_epoch=400,
                          status="continuation_training_success" if kernel == 3 else "formal_training_success",
                          checkpoint_sha256=file_digest(run / "model.pt"))
            if kernel == 3:
                phase = run / "source"
                phase.mkdir()
                for f in ("model.pt", "epoch300.pt", "config.resolved.yaml"):
                    (phase / f).write_text(f)
                (phase / "history.csv").write_text("epoch,train_loss,val_bands_1_10_mre_pct,val_bands_2_10_mre_pct\n" +
                                                    "".join(f"{i},1,1,1\n" for i in range(1, 301)))
                (phase / "results.json").write_text(json.dumps(dict(stop_epoch=300, best_epoch=1)))
                result["warm_start"] = dict(source_scientific_contract_verified=True, optimizer_state_restored=False,
                    source_epoch=300, initial_weights_role="best_validation", initial_weights_epoch=1, source_run=str(phase),
                    source_best_checkpoint=str(phase / "model.pt"), source_best_checkpoint_sha256=file_digest(phase / "model.pt"),
                    source_terminal_checkpoint=str(phase / "epoch300.pt"), source_terminal_checkpoint_sha256=file_digest(phase / "epoch300.pt"),
                    source_config_sha256=file_digest(phase / "config.resolved.yaml"))
            else:
                for f in ("phase300_best.pt", "epoch300.pt"):
                    (run / f).write_text(f)
                (run / "phase300_evidence.json").write_text(json.dumps(dict(
                    selected_checkpoint_sha256=file_digest(run / "phase300_best.pt"),
                    terminal_checkpoint_sha256=file_digest(run / "epoch300.pt"))))
                (run / "adaptive_phase_start.json").write_text(json.dumps(dict(
                    source_checkpoint_sha256=file_digest(run / "phase300_best.pt"),
                    optimizer_reset=True, random_states_reset=True, cumulative_epoch=301)))
            (run / "results.json").write_text(json.dumps(result))
            manifest["candidates"].append(dict(domain="domain", kernel=kernel, seed=seed, run_dir=str(run),
                                                checkpoint_sha256=result["checkpoint_sha256"], train_subset_hash="subset"))
    assert len(audit(manifest, source, protocol)["records"]) == 6
    (tmp_path / "3_0/source/epoch300.pt").write_text("changed")
    with pytest.raises(ValueError, match="source phase checkpoint changed"):
        audit(manifest, source, protocol)
