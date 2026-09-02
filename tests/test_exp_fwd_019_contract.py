"""Contract tests for EXP-FWD-019 component attribution.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from pathlib import Path

import pytest
import yaml

from phc2d.models import build_model, get_model_spec
from phc2d.forward.component_campaign import freeze_seed0_screen


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


def test_seed0_submitter_uses_three_way_n40_concurrency() -> None:
    submitter = (ROOT / "jobs/paracloud_submit_exp_fwd_019_seed0.sh").read_text(encoding="utf-8")
    worker = (ROOT / "jobs/paracloud_exp_fwd_019_seed0_train.sh").read_text(encoding="utf-8")
    assert "--array=0-20%3" in submitter
    assert "--gres=gpu:1" in submitter and "--mem=60000" in submitter
    assert "PHC2D_AUTHORIZE_EXP_FWD_019" in submitter
    assert "full_cell_dual_lane" in worker and "full_cell --config" in worker
