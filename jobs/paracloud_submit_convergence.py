"""Submit one eight-worker admission/train pair with an explicit dependency gate.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def submit(root: Path, suite: Path, stage: str, dependency: str | None) -> dict:
    profile = yaml.safe_load((suite / "config.resolved.yaml").read_text())
    record = dict(stage=stage, suite=str(suite), root=str(root),
                  submitted_at_utc=datetime.now(timezone.utc).isoformat(),
                  authorization_id=profile["authorization_id"], physical_gpu_limit=8,
                  slots=8, automatic_retry=False, dependency=dependency)
    marker = suite / f"submission_{stage}.json"
    with marker.open("x") as handle:
        json.dump(dict(record, status="submission_intent"), handle, indent=2)
    export = dict(PHC2D_ROOT=str(root), PHC2D_SUITE=str(suite), PHC2D_STAGE=stage,
                  PHC2D_GPU_PLATFORM=profile["platform"], PHC2D_ENV=profile["environment"],
                  PYTHONDONTWRITEBYTECODE="1")
    args = ["sbatch", "--parsable", "--no-requeue", "-p", profile["partition"],
            "--gres=gpu:1", "-c", str(profile["cpus_per_gpu"]),
            "-t", "01:00:00" if stage == "admission" else "7-00:00:00",
            "-J", "phc2d_convergence_" + stage, "--array=0-7%8",
            "-o", str(suite / (stage + "_%A_%a.out")), "-e", str(suite / (stage + "_%A_%a.err")),
            "--export=ALL," + ",".join(k + "=" + v for k, v in export.items())]
    if dependency:
        args.extend(["--dependency=afterok:" + dependency, "--kill-on-invalid-dep=yes"])
    args.append(str(root / "jobs/paracloud_convergence.sh"))
    completed = subprocess.run(args, capture_output=True, text=True, cwd=root)
    record.update(args=args, returncode=completed.returncode, stdout=completed.stdout,
                  stderr=completed.stderr, status="submitted" if completed.returncode == 0 else "submission_failed")
    if completed.returncode == 0:
        record["job_id"] = completed.stdout.strip().split(";")[0]
    marker.write_text(json.dumps(record, indent=2) + "\n")
    if completed.returncode:
        raise RuntimeError(record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    args = parser.parse_args()
    admission = submit(args.root, args.suite, "admission", None)
    print(json.dumps(admission), flush=True)
    print(json.dumps(submit(args.root, args.suite, "train", admission["job_id"])), flush=True)
