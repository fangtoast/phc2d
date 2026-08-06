"""Tests for full-cell forward cluster job helpers.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_ablation_status_parses_ablation_manifest(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sacct = fake_bin / "sacct"
    fake_sacct.write_text(
        "#!/usr/bin/env bash\nprintf 'COMPLETED|0:0|00:01:02|g0001|\\n'\n",
        encoding="utf-8",
    )
    fake_sacct.chmod(0o755)
    manifest = tmp_path / "submission_manifest.csv"
    manifest.write_text(
        "lane,representation,config,model,threshold,resolution,seed,job_id,run_dir,mode\n"
        "binary,binary-native,cfg.yaml,fno2d_band_v2,,32,0,12345,missing-run,train\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "jobs" / "paracloud_full_cell_forward_ablation_status.sh"),
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "binary,binary-native,fno2d_band_v2,32,0,12345" in completed.stdout
    assert "COMPLETED,0:0,00:01:02,g0001,missing" in completed.stdout


def test_ablation_status_rejects_main_wave_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.csv"
    manifest.write_text(
        "model,seed,job_id,run_dir,mode\nmlp,0,12345,run,train\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "jobs" / "paracloud_full_cell_forward_ablation_status.sh"),
            str(manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "status=invalid_ablation_manifest_header" in completed.stdout


def test_ablation_submitter_supports_selection_chunks(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "counter_file=sbatch-counter\n"
        "counter=100\n"
        "if [[ -f $counter_file ]]; then counter=$(<\"$counter_file\"); fi\n"
        "counter=$((counter + 1))\n"
        "printf '%s\\n' \"$counter\" > \"$counter_file\"\n"
        "printf '%s\\n' \"$counter\"\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    for name in ("binary.yaml", "continuous.yaml", "threshold.yaml"):
        (tmp_path / name).write_text("placeholder: true\n", encoding="utf-8")
    selection = tmp_path / "selection.csv"
    selection.write_text(
        "lane,representation,config,model,threshold,rest\n"
        "binary,binary,binary.yaml,fno2d_band_v2,,x\n"
        "graded,continuous,continuous.yaml,fno2d_band_v2,,x\n"
        "graded,t06,threshold.yaml,fno2d_band_v2,0.6,x\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "PHC2D_ABLATION_SELECTION_OFFSET": "2",
            "PHC2D_ABLATION_SELECTION_LIMIT": "1",
        }
    )

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "jobs" / "paracloud_submit_full_cell_forward_ablation.sh"),
            "train",
            str(selection),
            "chunk_test",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    manifest = tmp_path / completed.stdout.strip()
    rows = manifest.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 7
    assert all(row.startswith("graded,t06,") for row in rows[1:])
    assert all(",t06," in row for row in rows[1:])
