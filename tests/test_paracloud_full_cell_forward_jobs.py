"""Tests for full-cell forward cluster job helpers.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys


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


def test_main_wave_submitter_schedules_slowest_models_first(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "counter_file=sbatch-counter\n"
        "counter=200\n"
        "if [[ -f $counter_file ]]; then counter=$(<\"$counter_file\"); fi\n"
        "counter=$((counter + 1))\n"
        "printf '%s\\n' \"$counter\" > \"$counter_file\"\n"
        "printf '%s\\n' \"$counter\"\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    data = tmp_path / "dataset.h5"
    data.write_bytes(b"frozen-dataset")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    fake_env = tmp_path / "env"
    (fake_env / "bin").mkdir(parents=True)
    fake_python = fake_env / "bin" / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s|%s\\n' \"$FAKE_DATA_PATH\" \"$FAKE_DATA_SHA\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    config = tmp_path / "config.yaml"
    config.write_text("dataset: {}\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "PHC2D_ENV": str(fake_env),
            "FAKE_DATA_PATH": str(data),
            "FAKE_DATA_SHA": digest,
        }
    )

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "jobs" / "paracloud_submit_full_cell_forward_wave.sh"),
            "train",
            str(config),
            "ordered_wave",
            "",
            "64",
            "40000",
            "20",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    rows = (tmp_path / completed.stdout.strip()).read_text(encoding="utf-8").splitlines()[1:]
    assert len(rows) == 12
    assert all(row.startswith("fno2d_band_v2,") for row in rows[:3])
    assert all(row.startswith("lifted_cnn,") for row in rows[3:6])
    assert all(row.startswith("mlp,") for row in rows[-3:])


def test_forward_wrapper_defers_redundant_cuda_import_to_runner() -> None:
    wrapper = (ROOT / "jobs" / "paracloud_full_cell_forward.sh").read_text(encoding="utf-8")
    common = (ROOT / "jobs" / "paracloud_gpu_common.sh").read_text(encoding="utf-8")
    export_position = wrapper.index("export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1")
    source_position = wrapper.index('source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"')

    assert export_position < source_position
    assert 'if [[ "${PHC2D_DEFER_CUDA_CHECK_TO_RUNNER:-0}" != "1" ]]' in common
    assert "assert torch.cuda.is_available()" in common


def test_parameter_ablation_submitter_uses_bounded_gpu_array(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    args_file = tmp_path / "sbatch-args"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$FAKE_SBATCH_ARGS\"\nprintf '54321\\n'\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    manifest = tmp_path / "stage_a_manifest.csv"
    manifest.write_text(
        "stage,case_id,lane,model,parameter,value_json,is_current,seed,max_epochs,admission_probe,derived_config,run_dir\n"
        + "".join(
            f"a,c{i},binary,mlp,current_contract,null,true,0,1,true,cfg.yaml,run-{i}\n"
            for i in range(17)
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "PHC2D_ROOT": str(ROOT),
            "PHC2D_ENV": str(Path(sys.executable).resolve().parents[1]),
            "PHC2D_AUTHORIZE_PARAMETER_ABLATION": "YES",
            "FAKE_SBATCH_ARGS": str(args_file),
        }
    )

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "jobs" / "paracloud_submit_full_cell_parameter_ablation.sh"),
            str(manifest),
            "3",
            "2",
            "15",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "status=submitted job_id=54321 total_cases=17 offset=2 submitted_cases=15 concurrency=3" in completed.stdout
    args = args_file.read_text(encoding="utf-8")
    assert "--array=0-14%3" in args
    assert "PHC2D_PARAMETER_INDEX_OFFSET=2" in args


def test_parameter_ablation_worker_reuses_common_cuda_setup() -> None:
    worker = (ROOT / "jobs" / "paracloud_full_cell_parameter_ablation.sh").read_text(encoding="utf-8")
    defer_position = worker.index("export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1")
    source_position = worker.index('source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"')
    run_position = worker.index("python -m phc2d.forward.full_cell_parameter_ablation run-case")

    assert defer_position < source_position < run_position
