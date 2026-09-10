"""Contract tests for EXP-FWD-019 component attribution.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import csv
import json
from pathlib import Path

import pytest
import yaml

from phc2d.models import build_model, get_model_spec
from phc2d.forward.component_campaign import freeze_accepted_seed0, freeze_confirmation, freeze_recovery, freeze_seed0_screen


ROOT = Path(__file__).resolve().parents[1]


def test_controller_freezes_validation_only_funnel_and_n40_resources() -> None:
    config = yaml.safe_load((ROOT / "configs/research/full_cell_component_attribution_exp_fwd_019_v1.yaml").read_text(encoding="utf-8"))
    assert config["experiment"]["experiment_id"] == "EXP-FWD-019"
    assert config["protocol"]["c0"]["test_targets_loaded"] is False
    assert config["protocol"]["seed0_screen"] == {
        "seed": 0, "max_epochs": 20, "max_candidates_per_family": 2,
        "selection_metric": "validation_bands_2_10_mre",
    }
    runtime = config["protocol"]["runtime"]
    assert runtime["partition"] == "gpu_4090"
    assert runtime["gpus_per_task"] == 1 and runtime["cpus_per_task"] == 6
    assert runtime["max_concurrent_jobs"] == 3


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("fno2d_band_component_v1", {"use_band_gate": False}),
        ("tfno2d_band_component_v1", {"enabled_branches": ("spectral", "channel")}),
        ("convnext_band_component_v1", {"padding_mode": "zeros"}),
        ("dual_lane_band_component_v1", {"enabled_branches": ("local", "channel")}),
    ],
)
def test_component_identities_build_with_distinct_state_schema(name: str, params: dict[str, object]) -> None:
    torch = pytest.importorskip("torch")
    spec = get_model_spec(name)
    model = build_model(name, input_shape=(64, 64), output_shape=(81, 10), **params)
    kwargs = {"lane_id": torch.tensor([0, 1])} if spec.accepts_lane_id else {}
    if spec.accepts_k_points:
        kwargs["k_points"] = torch.rand(81, 2)
    assert model(torch.randn(2, 64, 64), **kwargs).shape == (2, 81, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_frozen_masks_preserve_existing_checkpoint_schema() -> None:
    pytest.importorskip("torch")
    for name, disabled in (
        ("fno2d_band_v2", ("band_gate",)),
        ("tfno2d_band_v3", ("local",)),
        ("dual_lane_band_operator_v1", ("spectral",)),
    ):
        model = build_model(name, input_shape=(64, 64), output_shape=(81, 10))
        before = tuple(model.state_dict())
        model.set_component_mask(disabled)
        assert tuple(model.state_dict()) == before


def test_recovery_freezer_preserves_config_hashes_and_uses_new_run_paths(tmp_path: Path) -> None:
    screen = tmp_path / "screen"; configs = screen / "derived_configs"; configs.mkdir(parents=True)
    rows = []
    for index in range(3):
        config = configs / f"case_{index}.yaml"
        config.write_text(f"seed: {index}\n", encoding="utf-8")
        rows.append({
            "task_index": index, "case_id": f"case_{index}", "family": "fno",
            "lane": "binary", "variant": "baseline", "model": "fno2d_band_component_v1",
            "seed": 0, "config": str(config), "run_dir": str(screen / "runs" / f"case_{index}"),
        })
    with (screen / "task_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    recovery = freeze_recovery(screen, tmp_path / "recovery", task_indices=(0, 2), repository_root=ROOT)
    with (recovery / "task_manifest.csv").open(encoding="utf-8", newline="") as handle:
        recovered = list(csv.DictReader(handle))
    with (recovery / "source_to_replacement_manifest.csv").open(encoding="utf-8", newline="") as handle:
        lineage = list(csv.DictReader(handle))
    assert [row["task_index"] for row in recovered] == ["0", "1"]
    assert [row["source_task_index"] for row in lineage] == ["0", "2"]
    assert all(row["source_config_sha256"] == row["replacement_config_sha256"] for row in lineage)
    assert all("recovery/runs" in row["run_dir"].replace("\\", "/") for row in recovered)


def test_accepted_seed0_derivative_combines_original_and_replacement_without_overwrite(tmp_path: Path) -> None:
    screen = tmp_path / "screen"; screen.mkdir()
    replacement = tmp_path / "replacement"; replacement.mkdir()
    rows = []
    for index in range(2):
        config = screen / f"case_{index}.yaml"; config.write_text("frozen: true\n", encoding="utf-8")
        run_dir = screen / "runs" / f"case_{index}"
        rows.append({"task_index": index, "case_id": f"case_{index}", "family": "fno", "lane": "binary", "variant": "baseline", "model": "fno2d_band_component_v1", "seed": 0, "config": str(config), "run_dir": str(run_dir)})
    with (screen / "task_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (screen / "selection.json").write_text('{"selected_candidates":{}}', encoding="utf-8")
    recovery = freeze_recovery(screen, replacement / "slice", task_indices=(0,), repository_root=ROOT)
    with (recovery / "task_manifest.csv").open(encoding="utf-8", newline="") as handle:
        recovery_row = next(csv.DictReader(handle))
    accepted_sources = [recovery_row, rows[1]]
    for row in accepted_sources:
        run_dir = Path(row["run_dir"]); run_dir.mkdir(parents=True)
        identity = {"requested_model_name": row["model"]}
        (run_dir / "results.json").write_text(json.dumps({"status": "formal_training_success", "formal_training_started": True, "test_executed": False, "model": row["model"], "model_identity": identity}), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps({"model_identity": identity}), encoding="utf-8")
        (run_dir / "run.meta.json").write_text("{}", encoding="utf-8")
        (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump({"experiment": {"component_family": row["family"], "component_variant": row["variant"], "seed": int(row["seed"])}}), encoding="utf-8")
        (run_dir / "model.pt").write_bytes(b"checkpoint")
    accepted = freeze_accepted_seed0(screen, (recovery,), tmp_path / "accepted", repository_root=ROOT)
    with (accepted / "task_manifest.csv").open(encoding="utf-8", newline="") as handle:
        accepted_rows = list(csv.DictReader(handle))
    assert len(accepted_rows) == 2
    assert accepted_rows[0]["run_dir"] == recovery_row["run_dir"]
    assert accepted_rows[1]["run_dir"] == rows[1]["run_dir"]
    assert json.loads((accepted / "acceptance.json").read_text(encoding="utf-8"))["test_targets_loaded"] is False


def test_c0_submitter_is_authorization_hash_and_concurrency_gated() -> None:
    submitter = (ROOT / "jobs/paracloud_submit_exp_fwd_019_c0.sh").read_text(encoding="utf-8")
    worker = (ROOT / "jobs/paracloud_exp_fwd_019_c0.sh").read_text(encoding="utf-8")
    assert "PHC2D_AUTHORIZE_EXP_FWD_019" in submitter
    assert "PHC2D_EXPECTED_FWD019_AUTHORIZATION_ID" in submitter
    assert "DEPLOYED_GIT_COMMIT" in submitter
    assert "--array=0-6%3" in submitter
    assert "--gres=gpu:1" in submitter and "-c 6" in submitter
    assert "component_attribution" in worker
    assert "SLURM_ARRAY_TASK_ID" in worker


def test_seed0_freezer_selects_two_candidates_and_writes_21_tasks(tmp_path: Path) -> None:
    c0 = tmp_path / "c0"; c0.mkdir()
    (c0 / "c0_manifest.csv").write_text("task_index\n0\n", encoding="utf-8")
    masks = {
        "fno_b": ("primary_spectral", "band_spectral", "band_gate", "low_rank"),
        "fno_g": ("primary_spectral", "band_spectral", "band_gate", "low_rank"),
        "tfno_b": ("spectral", "local", "channel"),
        "tfno_g": ("spectral", "local", "channel"),
        "joint": ("spectral", "local", "channel"),
    }
    for case in ("fno_b", "fno_g", "tfno_b", "tfno_g", "convnext_b", "convnext_g", "joint"):
        case_dir = c0 / case; case_dir.mkdir()
        (case_dir / "results.json").write_text('{"status":"frozen_validation_diagnostic_success","test_targets_loaded":false}', encoding="utf-8")
        lane_name = "binary" if case.endswith("_b") else ("graded" if case.endswith("_g") else "joint")
        perturbations = {
            f"mask_{mask}": {"prediction_delta_relative_pct": float(index + 1)}
            for index, mask in enumerate(masks.get(case, ()))
        }
        (case_dir / "metrics.json").write_text(
            __import__("json").dumps({"lanes": {lane_name: {"perturbations": perturbations}}}),
            encoding="utf-8",
        )
    output = freeze_seed0_screen(c0, tmp_path / "screen", repository_root=ROOT)
    rows = (output / "task_manifest.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 22
    selection = yaml.safe_load((output / "selection.json").read_text(encoding="utf-8"))
    assert all(len(values) == 2 for values in selection["selected_candidates"].values())
    assert selection["test_targets_loaded"] is False
    with (output / "task_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    for row in manifest:
        run_dir = Path(row["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(
            json.dumps({"status": "dual_lane_training_success" if row["lane"] == "joint" else "formal_training_success", "test_executed": False}),
            encoding="utf-8",
        )
        value = 1.0 if row["variant"] == "baseline" else 1.1
        metrics = (
            {"binary_validation_bands_2_10_mre_pct": value, "graded_validation_bands_2_10_mre_pct": value}
            if row["lane"] == "joint" else {"bands_2_10_mre_pct": value}
        )
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    confirmation = freeze_confirmation(output, tmp_path / "confirmation", repository_root=ROOT)
    assert len((confirmation / "task_manifest.csv").read_text(encoding="utf-8").splitlines()) == 43
    gate = json.loads((confirmation / "candidate_gate.json").read_text(encoding="utf-8"))
    assert gate["task_count"] == 42
    assert set(gate["selected_variants"]) == {"fno", "tfno", "convnext", "joint"}


def test_seed0_submitter_uses_three_way_n40_concurrency() -> None:
    submitter = (ROOT / "jobs/paracloud_submit_exp_fwd_019_seed0.sh").read_text(encoding="utf-8")
    worker = (ROOT / "jobs/paracloud_exp_fwd_019_seed0_train.sh").read_text(encoding="utf-8")
    chunk_lib = (ROOT / "jobs/paracloud_exp_fwd_019_chunk_lib.sh").read_text(encoding="utf-8")
    assert 'end - start + 1 <= 16' in chunk_lib
    assert '--array="${start}-${end}%3"' in chunk_lib
    assert "prior chunk" in chunk_lib and 'states" == "COMPLETED"' in chunk_lib
    assert "submission_chunk_" in chunk_lib
    assert "--gres=gpu:1" in chunk_lib and "-c 6" in chunk_lib
    assert "--mem=" not in chunk_lib
    assert "PHC2D_AUTHORIZE_EXP_FWD_019" in submitter
    assert "full_cell_dual_lane" in worker and "full_cell --config" in worker
    confirmation = (ROOT / "jobs/paracloud_submit_exp_fwd_019_confirmation.sh").read_text(encoding="utf-8")
    assert 'phc2d_fwd019_submit_chunk "$campaign" "$start" "$end" "$count"' in confirmation
    assert "three_seed_validation_only_no_test" in confirmation
    recovery = (ROOT / "jobs/paracloud_submit_exp_fwd_019_recovery.sh").read_text(encoding="utf-8")
    assert "--recovery-screen-dir" in recovery and "--task-indices" in recovery
    assert "source_task_indices" in recovery and "seed0_validation_only_no_test_correction" in recovery
    assert "PHC2D_EXPECTED_SOURCE_JOB_ID" in recovery and 'state" == "FAILED"' in recovery
    continuation = (ROOT / "jobs/paracloud_submit_exp_fwd_019_unattempted.sh").read_text(encoding="utf-8")
    assert "source_index > source_end" in continuation
    assert '! -e "$source_run_dir"' in continuation
    assert "seed0_validation_only_no_test_unattempted" in continuation
