"""Submit EXP-FWD-031 test cells as training frees GPU capacity.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

EXTERNAL = {
    "dat007_12000": ("exp_dat_007_single_hole_six_family_bz9_12000_bingxing_v1.h5",
                     "badd2081180406d6eaac451d758c8d17259a23e13a2c00198b2c82059c9b85a8"),
    "dat008_24000": ("exp_dat_008_pcsel_twelve_family_bz9_24000_bingxing_v1.h5",
                     "8c3721e03603a01dd5a8e5c8f6b183e8c70259a259e18ac2f53483585e04aa5c"),
}


def gpu_jobs() -> int:
    result = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%b"],
                            capture_output=True, text=True, check=True)
    return sum("gpu:" in line or "gres/gpu:" in line for line in result.stdout.splitlines())


def submitted_job_state(claim: Path) -> str:
    record_path = claim / ("replacement.json" if (claim / "replacement.json").is_file() else "submission.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    job_id = record["stdout"].strip().split(";")[0]
    result = subprocess.run(["sacct", "-j", job_id, "--noheader", "--format=State"],
                            capture_output=True, text=True, check=True)
    return result.stdout.splitlines()[0].strip().split()[0] if result.stdout.strip() else "UNKNOWN"


def targets(profile: dict, external_root: Path) -> list[tuple[str, Path, str, str]]:
    cells = []
    for name, item in profile["datasets"].items():
        cells.append((name, Path(profile["dataset_root"]) / Path(item["path"]).name,
                      item["sha256"], item["grid_field"]))
    for name, (filename, digest) in EXTERNAL.items():
        cells.append((name, external_root / filename, digest, "material_area_fraction_grid_64"))
    return cells


def scan(suite: Path, worker: Path, source: Path, deployment: Path,
         archive_sha256: str, external_root: Path) -> dict:
    profile = yaml.safe_load((suite / "config.resolved.yaml").read_text(encoding="utf-8"))
    if profile["experiment_id"] != "EXP-FWD-031" or profile["gpu_limit"] != 8:
        raise ValueError("unexpected campaign GPU contract")
    platform = profile["platform"]
    with (suite / "task_manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 18 or len({row["case_id"] for row in rows}) != 18:
        raise ValueError("unexpected local training manifest")
    desired = targets(profile, external_root)
    # The independent validation dispatcher can enqueue one GPU job between
    # our capacity check and sbatch. Keep one slot free until its 18 outputs
    # and all 18 formal training runs are finished on this server.
    training_done = sum((suite / "train_claims" / f"{suite.name}_{row['case_id']}" /
                         "completed.json").is_file() for row in rows)
    validation_done = sum((suite / "training" / row["case_id"] /
                           "per_band_validation.json").is_file() for row in rows)
    local_limit = 8 if training_done == 18 and validation_done == 18 else 7
    capacity = max(0, local_limit - gpu_jobs())
    submitted = 0
    finished = 0
    blocked = 0
    # Native cells first, then cross-domain, then new-geometry transfer.
    candidates = [(row, target) for row in rows for target in desired]
    candidates.sort(key=lambda item: (0 if item[0]["case_id"].startswith(item[1][0] + "_") else
                                      2 if item[1][0].startswith("dat") else 1,
                                      item[0]["task_index"], item[1][0]))
    for row, (name, data, digest, grid) in candidates:
        case = row["case_id"]
        completed = suite / "train_claims" / f"{suite.name}_{case}" / "completed.json"
        failed_train = completed.with_name("failure.json")
        output = suite / "test" / case / name
        claim = suite / "test_claims" / case / name
        if all((output / file).is_file() for file in
               ("results.json", "metrics.json", "per_sample_metrics.csv", "per_band_metrics.csv")):
            finished += 1
            continue
        if failed_train.is_file() or (claim / "failure.json").is_file() or output.exists():
            blocked += 1
            continue
        if claim.exists():
            if (claim / "submission.json").is_file() and submitted_job_state(claim) in {
                    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
                (claim / "failure.json").write_text(json.dumps(dict(
                    reason="submitted job ended without complete test artifacts")) + "\n", encoding="utf-8")
                blocked += 1
            continue
        validation_record = suite / "training" / case / "per_band_validation.json"
        if not completed.is_file() or not validation_record.is_file() or capacity <= 0:
            continue
        training_result = json.loads((suite / "training" / case / "results.json").read_text(encoding="utf-8"))
        if training_result["status"] not in {"formal_training_success", "training_complete_convergence_not_met"}:
            blocked += 1
            continue
        if not data.is_file():
            blocked += 1
            continue
        claim.mkdir(parents=True, exist_ok=False)
        env = ",".join(f"{key}={value}" for key, value in {
            "PHC2D_SUITE": suite, "PHC2D_CASE": case, "PHC2D_TARGET": name,
            "PHC2D_DATASET": data, "PHC2D_DATASET_SHA256": digest,
            "PHC2D_GRID_FIELD": grid, "PHC2D_TEST_SOURCE": source,
            "PHC2D_DEPLOY_ROOT": deployment, "PHC2D_ARCHIVE_SHA256": archive_sha256,
            "PHC2D_GPU_PLATFORM": platform}.items())
        command = ["sbatch", "--parsable", "--no-requeue", "-p", profile["partition"],
                   "--gres=gpu:1", "-c", str(profile["cpus_per_gpu"]), "--time=04:00:00",
                   "-J", f"phc2d031_test_{platform}", f"--export=ALL,{env}",
                   f"--output={claim / 'job_%j.out'}", f"--error={claim / 'job_%j.err'}", str(worker)]
        submission = subprocess.run(command, capture_output=True, text=True)
        record = dict(command=command, returncode=submission.returncode,
                      stdout=submission.stdout, stderr=submission.stderr)
        (claim / ("submission.json" if submission.returncode == 0 else "failure.json")).write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if submission.returncode:
            blocked += 1
        else:
            capacity -= 1
            submitted += 1
    return dict(expected=len(candidates), complete=finished, blocked=blocked, submitted=submitted,
                available_gpu_slots=capacity)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    for name in ("suite", "worker", "source", "deployment", "external-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        state = scan(args.suite, args.worker, args.source, args.deployment,
                     args.archive_sha256, args.external_root)
        print(json.dumps(state), flush=True)
        if not args.watch or state["complete"] + state["blocked"] == state["expected"]:
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
