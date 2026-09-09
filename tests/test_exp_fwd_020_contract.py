"""Contract tests for EXP-FWD-020 global-local bridge.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from pathlib import Path
import csv

import pytest
import yaml

from phc2d.models import build_model, get_model_spec
from phc2d.forward.global_local_bridge_campaign import freeze_g1


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
    assert config["runtime"]["partition"] == "gpu_5090"
    assert config["runtime"]["torch_version"] == "2.7.1"
    assert config["runtime"]["max_concurrent_jobs"] == 8
    assert len(config["datasets"]) == 6
    assert all("n56_path" in item for item in config["datasets"].values())

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
    assert str(resolved["dataset"]["path"]).startswith("/data/home/scxl892/")


def test_exp_fwd_020_submitter_requires_and_propagates_versioned_environment() -> None:
    submitter = (ROOT / "jobs/paracloud_submit_exp_fwd_020.sh").read_text(encoding="utf-8")
    assert ': "${PHC2D_ENV:?Set the versioned runtime environment directory.}"' in submitter
    assert '"$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_bridge_campaign' in submitter
    assert 'PHC2D_ENV=$PHC2D_ENV' in submitter
    assert "$HOME/run/envs/phc2d_gpu_cu128_torch271/bin/python" not in submitter
