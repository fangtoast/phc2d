"""Tests for the authorized full-cell 50k rolling training campaign.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from phc2d.forward.full_cell_50k_campaign import (
    MODELS,
    advance_campaign,
    audit_wave,
    build_wave_plan,
    initialize_campaign,
)


def _write_manifest(
    root: Path,
    name: str,
    *,
    create_artifacts: bool,
    epochs: int = 20,
    train_size: int = 40000,
    final_fit: bool = False,
) -> Path:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for seed in (0, 1, 2):
            run_dir = root / "runs" / f"{name}_{model}_seed{seed}"
            if create_artifacts:
                (run_dir / "figures").mkdir(parents=True)
                (run_dir / "run.log").write_text("success\n", encoding="utf-8")
                (run_dir / "model.pt").write_bytes(b"checkpoint")
                (run_dir / "figures" / "training_curve.png").write_bytes(b"png")
                (run_dir / "figures" / "training_curve.pdf").write_bytes(b"pdf")
                (run_dir / "results.json").write_text(
                    json.dumps(
                        {
                            "status": "formal_training_success",
                            "test_executed": False,
                            "selection_split": "none_final_fit" if final_fit else "validation",
                            "checkpoint_role": (
                                "final_epoch_final_fit" if final_fit else "best_validation"
                            ),
                            "validation_overlap": final_fit,
                            "validation_metrics_admissible": not final_fit,
                            "model": model,
                            "seed": seed,
                            "train_size": train_size,
                            "max_epochs": epochs,
                            "checkpoint_sha256": "a" * 64,
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "best_validation_bands_2_10_mre_pct": None if final_fit else 1.0,
                            "best_training_overlap_monitor_mre_pct": 1.0 if final_fit else None,
                            "monitor_is_unbiased_validation": not final_fit,
                            "best_epoch": epochs if final_fit else 1,
                            "checkpoint_epoch": epochs if final_fit else 1,
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "run.meta.json").write_text(
                    json.dumps(
                        {
                            "dataset_sha256": "b" * 64,
                            "train_subset_hash": "c" * 64,
                            "full_dataset_final_fit": final_fit,
                            "validation_overlap_count": 5000 if final_fit else 0,
                            "test_overlap_count": 5000 if final_fit else 0,
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "config.resolved.yaml").write_text(
                    yaml.safe_dump(
                        {"training": {"train_size": train_size, "max_epochs": epochs}}
                    ),
                    encoding="utf-8",
                )
                with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["epoch", "val_bands_2_10_mre_pct"])
                    writer.writeheader()
                    for epoch in range(1, epochs + 1):
                        writer.writerow({"epoch": epoch, "val_bands_2_10_mre_pct": 1.0})
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "job_id": 1000 + len(rows),
                    "run_dir": str(run_dir.relative_to(root)),
                    "mode": "train",
                }
            )
    manifest = root / "runs" / name / "submission_manifest.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _states(state: str) -> dict[str, dict[str, str]]:
    return {str(1000 + index): {"state": state} for index in range(12)}


def test_build_wave_plan_freezes_17_waves_and_204_runs() -> None:
    waves = build_wave_plan()
    assert len(waves) == 17
    assert sum(wave["planned_runs"] for wave in waves) == 204
    assert waves[0]["key"] == "binary_native_n40000_e20"
    assert waves[1]["key"] == "binary_native_n40000_e100"
    assert waves[2]["key"] == "binary_native_n40000_e200"
    assert waves[3]["key"] == "graded_continuous_n40000_e20"
    assert waves[6]["key"] == "binary_native_n50000_e20"
    assert waves[12]["key"] == "graded_t03_n50000_e100"
    assert waves[-1]["key"] == "graded_t07_n50000_e100"
    assert {int(wave["train_size"]) for wave in waves} == {40000, 50000}
    threshold_waves = [wave for wave in waves if str(wave["representation"]).startswith("graded_t")]
    assert len(threshold_waves) == 5
    assert {(wave["train_size"], wave["max_epochs"]) for wave in threshold_waves} == {
        (50000, 100)
    }


def test_final_fit_wave_audit_requires_explicit_overlap_boundary(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "final_fit",
        create_artifacts=True,
        epochs=100,
        train_size=50000,
        final_fit=True,
    )
    wave = next(wave for wave in build_wave_plan() if wave["key"] == "binary_native_n50000_e100")
    wave["manifest"] = str(manifest.relative_to(tmp_path))
    result = audit_wave(
        wave,
        repository_root=tmp_path,
        slurm_reader=lambda _ids: _states("COMPLETED"),
    )
    assert result == {
        "status": "complete",
        "completed": 12,
        "failed": 0,
        "active": 0,
        "unknown": 0,
        "failures": [],
    }


def test_advance_is_idempotent_and_submits_only_after_complete_audit(tmp_path: Path) -> None:
    first = _write_manifest(tmp_path, "first", create_artifacts=True)
    state_path = initialize_campaign(
        "campaign",
        repository_root=tmp_path,
        adopted_manifests={"binary_native_n40000_e20": first},
        authorization_id="user-goal",
    )
    submitted: list[str] = []

    def submitter(wave: dict[str, object]) -> Path:
        submitted.append(str(wave["key"]))
        return _write_manifest(tmp_path, "second", create_artifacts=False, epochs=100)

    def aggregator(_manifest: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True)
        (output_dir / "wave.meta.json").write_text(
            json.dumps({"status": "formal_wave_aggregation_success"}),
            encoding="utf-8",
        )

    state = advance_campaign(
        "campaign",
        repository_root=tmp_path,
        submit=True,
        slurm_reader=lambda _ids: _states("COMPLETED"),
        submitter=submitter,
        aggregator=aggregator,
    )
    assert state["completed_waves"] == 1
    assert state["completed_runs"] == 12
    assert state["active_wave"] == "binary_native_n40000_e100"
    assert submitted == ["binary_native_n40000_e100"]

    state = advance_campaign(
        "campaign",
        repository_root=tmp_path,
        submit=True,
        slurm_reader=lambda _ids: _states("PENDING"),
        submitter=submitter,
        aggregator=aggregator,
    )
    assert state["active_wave"] == "binary_native_n40000_e100"
    assert submitted == ["binary_native_n40000_e100"]
    assert json.loads(state_path.read_text(encoding="utf-8"))["completed_runs"] == 12


def test_failed_artifact_closes_campaign_without_automatic_retry(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, "broken", create_artifacts=True)
    broken_run = tmp_path / "runs" / "broken_mlp_seed0"
    (broken_run / "model.pt").unlink()
    initialize_campaign(
        "campaign",
        repository_root=tmp_path,
        adopted_manifests={"binary_native_n40000_e20": manifest},
        authorization_id="user-goal",
    )
    submitted: list[str] = []
    state = advance_campaign(
        "campaign",
        repository_root=tmp_path,
        submit=True,
        slurm_reader=lambda _ids: _states("COMPLETED"),
        submitter=lambda wave: submitted.append(str(wave["key"])),  # type: ignore[arg-type,return-value]
        aggregator=lambda _manifest, _output: None,
    )
    assert state["status"] == "blocked_failed_wave"
    assert state["failed_runs"] == 1
    assert submitted == []
    assert state["next_gate"] == "manual_failure_audit_no_automatic_retry"


def test_running_wave_never_submits_another_wave(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, "running", create_artifacts=False)
    initialize_campaign(
        "campaign",
        repository_root=tmp_path,
        adopted_manifests={"binary_native_n40000_e20": manifest},
        authorization_id="user-goal",
    )
    submitted: list[str] = []
    state = advance_campaign(
        "campaign",
        repository_root=tmp_path,
        submit=True,
        slurm_reader=lambda _ids: _states("RUNNING"),
        submitter=lambda wave: submitted.append(str(wave["key"])),  # type: ignore[arg-type,return-value]
        aggregator=lambda _manifest, _output: None,
    )
    assert state["completed_waves"] == 0
    assert state["active_wave"] == "binary_native_n40000_e20"
    assert submitted == []


def test_interrupted_submission_fails_closed_instead_of_duplicating(tmp_path: Path) -> None:
    state_path = initialize_campaign(
        "campaign",
        repository_root=tmp_path,
        authorization_id="user-goal",
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["waves"][0]["status"] = "submitting"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    submitted: list[str] = []
    state = advance_campaign(
        "campaign",
        repository_root=tmp_path,
        submit=True,
        slurm_reader=lambda _ids: {},
        submitter=lambda wave: submitted.append(str(wave["key"])),  # type: ignore[arg-type,return-value]
        aggregator=lambda _manifest, _output: None,
    )
    assert state["status"] == "blocked_submission_failure"
    assert state["submission_recovery_required"] == ["binary_native_n40000_e20"]
    assert submitted == []
