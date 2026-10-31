"""Submit one explicitly gated EXP-FWD-032 worker array.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from phc2d.forward.budget_confirmation import file_digest
from phc2d.forward.single_domain_branch_campaign import AUTHORIZATION

ENVIRONMENTS = {"n40": "/HOME/scw6g7a/run/envs/phc2d_gpu_cu118",
                "n56": "/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1"}


def submit(root: Path, suite: Path, stage: str, *, recovery_id: str | None = None,
           recovery_authorization: str | None = None) -> dict:
    profile = yaml.safe_load((suite / "config.resolved.yaml").read_text(encoding="utf-8"))
    platform = profile["platform"]
    if (profile["experiment_id"] != "EXP-FWD-032" or platform not in ENVIRONMENTS
            or profile["execution"]["authorization_id"] != AUTHORIZATION or stage not in {"admission", "train"}):
        raise ValueError("unregistered submission contract")
    if stage == "train":
        from phc2d.forward.single_domain_branch_campaign import audit_release
        audit_release(suite)
    deployment = json.loads((root / "deployment_audit.json").read_text())
    launcher = root / "jobs/paracloud_single_domain_branch.sh"
    launcher_audit = json.loads((root / "launcher_audit.json").read_text())
    if (launcher_audit["archive_sha256"] != deployment["archive_sha256"]
            or file_digest(launcher) != launcher_audit["launcher_sha256"]):
        raise ValueError("launcher is missing or does not match the frozen archive")
    platform_spec = profile["platforms"][platform]
    marker = suite / f"submission_{stage}.json"
    if recovery_id is not None:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", recovery_id) or not recovery_authorization:
            raise ValueError("a recovery identifier and explicit authorization are required")
        previous = [marker, *suite.glob(f"submission_{stage}_recovery_*.json")]
        for path in previous:
            record = json.loads(path.read_text())
            if record.get("status") != "submission_failed" or record.get("stdout", "").strip():
                raise ValueError("previous submission may have created a job; inspect manually")
        marker = suite / f"submission_{stage}_recovery_{recovery_id}.json"
    elif recovery_authorization is not None:
        raise ValueError("recovery authorization requires a recovery identifier")
    record = dict(status="submission_intent", stage=stage, authorization_id=AUTHORIZATION,
                  recovery_id=recovery_id, recovery_authorization=recovery_authorization,
                  submitter_sha256=file_digest(Path(__file__)), launcher_sha256=launcher_audit["launcher_sha256"],
                  deployment_archive_sha256=deployment["archive_sha256"],
                  submitted_at_utc=datetime.now(timezone.utc).isoformat(), automatic_retry=False)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    exports = dict(PHC2D_ROOT=str(root), PHC2D_SUITE=str(suite), PHC2D_STAGE=stage,
                   PHC2D_GPU_PLATFORM=platform, PHC2D_ENV=ENVIRONMENTS[platform],
                   PHC2D_ARCHIVE_SHA256=deployment["archive_sha256"], PYTHONDONTWRITEBYTECODE="1")
    args = ["sbatch", "--parsable", "--no-requeue", "-p", platform_spec["partition"],
            "--gres=gpu:1", "-c", str(platform_spec["cpus_per_gpu"]),
            "-t", "04:00:00" if stage == "admission" else "7-00:00:00",
            "-J", "phc2d032_" + stage, "--array=0-7%8",
            "-o", str(suite / (stage + "_%A_%a.out")), "-e", str(suite / (stage + "_%A_%a.err")),
            "--export=ALL," + ",".join(k + "=" + v for k, v in exports.items()),
            str(launcher)]
    result = subprocess.run(args, capture_output=True, text=True, cwd=root)
    record.update(command=args, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                  status="submitted" if result.returncode == 0 else "submission_failed")
    if result.returncode == 0:
        record["job_id"] = result.stdout.strip().split(";")[0]
    marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if result.returncode:
        raise RuntimeError(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=["admission", "train"])
    parser.add_argument("--recovery-id")
    parser.add_argument("--recovery-authorization")
    args = parser.parse_args()
    print(json.dumps(submit(args.root, args.suite, args.stage, recovery_id=args.recovery_id,
                            recovery_authorization=args.recovery_authorization)), flush=True)


if __name__ == "__main__":
    main()
