"""Contract tests for the Xingyi MPB resolution Stage-A scripts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_stage_a_config_freezes_the_authorized_matrix() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/research/full_cell_resolution_convergence_xingyi_stage_a_v1.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["experiment"]["phase"] == "stage_a_resource_pilot"
    assert config["selection"] == {
        "samples_per_lane": 8,
        "policy": "deterministic_round_robin_by_generator_stratum",
        "k_policy": "full_bz_all",
    }
    assert config["resolutions"] == [32, 48, 64, 96, 128]
    assert config["workers"]["workers"] == 16
    assert config["solver"]["environment"] == "native_linux_conda_meep_env"


def test_stage_a_submission_is_dry_run_and_authorization_gated() -> None:
    runner = (ROOT / "jobs/xingyi_full_cell_resolution_stage_a.sh").read_text(encoding="utf-8")
    submitter = (ROOT / "jobs/xingyi_submit_full_cell_resolution_stage_a.sh").read_text(encoding="utf-8")

    assert "PHC2D_AUTHORIZE_RESOLUTION_STAGE_A" in runner
    assert "SLURM_CPUS_PER_TASK" in runner
    assert "/usr/bin/time -v" in runner
    assert "resource_usage_source=slurm_sacct" in runner
    assert "if [[ -x /usr/bin/time ]]" in runner
    assert 'PHC2D_SUBMIT_MODE="${PHC2D_SUBMIT_MODE:-dry-run}"' in submitter
    assert submitter.index('if [[ "$PHC2D_SUBMIT_MODE" == "dry-run" ]]') < submitter.index(
        'job_id="$(sbatch --parsable'
    )
    assert "--time=08:00:00" in submitter


def test_stage_b_config_freezes_exact_100_structure_strata() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/research/full_cell_resolution_convergence_xingyi_stage_b_v1.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["experiment"]["phase"] == "stage_b_formal_confirmation"
    assert config["selection"]["samples_per_lane"] == 50
    assert config["selection"]["stratum_counts"]["binary_air_holes"] == {
        "binary_polar_holes_1": 13,
        "binary_polar_holes_2": 12,
        "binary_spectral_threshold": 25,
    }
    assert config["selection"]["stratum_counts"]["graded_medium"] == {
        "graded_core_1": 25,
        "graded_core_2": 25,
    }
    assert config["gates"]["recommendation_resolutions"] == [64, 96]


def test_stage_b_submission_is_dry_run_and_authorization_gated() -> None:
    runner = (ROOT / "jobs/xingyi_full_cell_resolution_stage_b.sh").read_text(encoding="utf-8")
    submitter = (ROOT / "jobs/xingyi_submit_full_cell_resolution_stage_b.sh").read_text(encoding="utf-8")

    assert "PHC2D_AUTHORIZE_RESOLUTION_STAGE_B" in runner
    assert "full_cell_resolution_convergence_xingyi_stage_b_v1.yaml" in runner
    assert 'PHC2D_SUBMIT_MODE="${PHC2D_SUBMIT_MODE:-dry-run}"' in submitter
    assert submitter.index('if [[ "$PHC2D_SUBMIT_MODE" == "dry-run" ]]') < submitter.index(
        'job_id="$(sbatch --parsable'
    )
    assert "--time=02:00:00" in submitter
