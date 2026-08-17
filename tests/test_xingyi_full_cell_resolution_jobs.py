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
    assert 'PHC2D_SUBMIT_MODE="${PHC2D_SUBMIT_MODE:-dry-run}"' in submitter
    assert submitter.index('if [[ "$PHC2D_SUBMIT_MODE" == "dry-run" ]]') < submitter.index(
        'job_id="$(sbatch --parsable'
    )
    assert "--time=08:00:00" in submitter
