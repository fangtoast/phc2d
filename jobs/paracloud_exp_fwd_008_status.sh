#!/usr/bin/env bash
# Read-only EXP-FWD-008 status check; watch mode samples every 30 minutes.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
campaign="${PHC2D_FWD008_CAMPAIGN_DIR:?Set PHC2D_FWD008_CAMPAIGN_DIR}"
mode="${1:-once}"
root="${PHC2D_ROOT:-$HOME/run/phc2d}"
python_bin="${PHC2D_PYTHON:-$HOME/run/envs/phc2d_gpu_cu118/bin/python}"

check_once() {
  date -Is
  if command -v squeue >/dev/null 2>&1; then squeue -u "${USER:-$(id -un)}" -o '%.18i %.20j %.10T %.10M %.20R' || true; fi
  if command -v sacct >/dev/null 2>&1 && [[ -f "$campaign/submission.env" ]]; then
    job_ids="$(awk -F= '/job_id=|shard_job_id=|finalizer_job_id=/{print $2}' "$campaign/submission.env" | paste -sd, -)"
    [[ -n "$job_ids" ]] && sacct -j "$job_ids" --format=JobID,JobName,State,Elapsed,ExitCode -n -P || true
  fi
  if [[ -f "$campaign/task_manifest.csv" ]]; then
    "$python_bin" - "$campaign" <<'PY'
import csv
import json
import math
import pathlib
import re
import sys

campaign = pathlib.Path(sys.argv[1])


def read_rows(name):
    path = campaign / name
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def unique_tasks():
    rows = []
    seen = set()
    for name in ("admission_manifest.csv", "task_manifest.csv"):
        for row in read_rows(name):
            key = (row.get("stage", ""), row.get("task_id", ""), row.get("run_dir", ""))
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def history_state(run_dir):
    history = run_dir / "history.csv"
    latest = "-"
    rows = 0
    nonfinite = False
    if history.is_file():
        try:
            with history.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows += 1
                    latest = row.get("epoch", latest)
                    for value in row.values():
                        if value in (None, ""):
                            continue
                        try:
                            if not math.isfinite(float(value)):
                                nonfinite = True
                        except ValueError:
                            pass
        except OSError:
            pass
    return latest, rows, nonfinite


def error_flags(run_dir, nonfinite):
    flags = []
    for name in ("run.log", "run.err", "stderr.log"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[-200000:].lower()
        except OSError:
            continue
        if "out of memory" in text or "cuda oom" in text:
            flags.append("OOM")
        if re.search(r"\bnonfinite\b|\bnan\b|\binf(?:inity)?\b", text):
            nonfinite = True
    if nonfinite:
        flags.append("NONFINITE")
    return "+".join(dict.fromkeys(flags)) or "-"


tasks = unique_tasks()
print("training_task_count=" + str(len(tasks)))
print("stage task_id wave lane latest_epoch history_rows best_epoch status history model epoch200 flags")
for row in tasks:
    run_dir = pathlib.Path(row.get("run_dir", ""))
    latest, history_rows, nonfinite = history_state(run_dir)
    result = load_json(run_dir / "results.json")
    metrics = load_json(run_dir / "metrics.json")
    summary = load_json(run_dir / "model_summary.json")
    source = {**summary, **metrics, **result}
    status = source.get("status", "not_started")
    best_epoch = source.get("best_epoch", source.get("best_validation_epoch", "-"))
    if status == "not_started" and history_rows:
        status = "training"
    flags = error_flags(run_dir, nonfinite)
    print(" ".join(str(value) for value in (
        row.get("stage", "-"), row.get("task_id", "-"), row.get("wave", "-"),
        row.get("train_lane", "-"), latest, history_rows, best_epoch, status,
        "yes" if (run_dir / "history.csv").is_file() else "no",
        "yes" if (run_dir / "model.pt").is_file() else "no",
        "yes" if (run_dir / "epoch200.pt").is_file() else "no", flags,
    )))
PY
  fi
  find "$campaign" -name results.json -o -name metrics.json -o -name history.csv -o -name epoch200.pt | wc -l | awk '{print "observed_artifact_files=" $1}'
}

case "$mode" in
  once) check_once ;;
  watch)
    while true; do check_once; sleep 1800; done
    ;;
  *) echo "usage: $0 once|watch" >&2; exit 2 ;;
esac
