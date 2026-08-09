#!/usr/bin/env bash
# Submit the authorized frozen-winner threshold diagnostic and finalizer.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

mode="${1:-}"
threshold_dir="${2:-}"
authorization_id="${3:-}"
if [[ "$mode" != "submit" || -z "$threshold_dir" || -z "$authorization_id" ]]; then
  echo "usage: $0 submit THRESHOLD_DIR AUTHORIZATION_ID" >&2
  exit 2
fi
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
cd "$PHC2D_ROOT"
actual_commit="$(tr -d '[:space:]' < DEPLOYED_GIT_COMMIT)"
expected_commit="${PHC2D_EXPECTED_COMMIT:-}"
if [[ -z "$expected_commit" || "$actual_commit" != "$expected_commit" ]]; then
  echo "deployment mismatch: deployed=$actual_commit expected=${expected_commit:-missing}" >&2
  exit 3
fi
if [[ "${PHC2D_AUTHORIZE_UPGRADE_50K_EVAL:-}" != "YES" || "$authorization_id" != "${PHC2D_EXPECTED_50K_EVAL_AUTHORIZATION_ID:-}" ]]; then
  echo "threshold diagnostic authorization mismatch" >&2
  exit 4
fi
for artifact in config.resolved.yaml threshold_plan.csv campaign_status.json; do
  test -s "$threshold_dir/$artifact" || { echo "threshold campaign missing $artifact" >&2; exit 5; }
done
if grep -q '^shard_job_id=' "$threshold_dir/submission.env" 2>/dev/null; then
  echo "threshold campaign was already submitted" >&2
  exit 5
fi
python3 - "$threshold_dir" <<'PY'
import hashlib
import json
import pathlib
import sys

import yaml

threshold = pathlib.Path(sys.argv[1])
config = yaml.safe_load((threshold / "config.resolved.yaml").read_text(encoding="utf-8"))
status = json.loads((threshold / "campaign_status.json").read_text(encoding="utf-8"))
if status.get("status") != "threshold_plan_ready" or status.get("planned_cells") != 15:
    raise SystemExit("threshold plan is not ready")
if hashlib.sha256((threshold / "threshold_plan.csv").read_bytes()).hexdigest() != config["plan_hash"]:
    raise SystemExit("threshold plan hash mismatch")
parent = pathlib.Path.cwd() / config["parent_campaign"]
result = json.loads((parent / "results.json").read_text(encoding="utf-8"))
if result.get("status") != "formal_upgrade_50k_evaluation_success":
    raise SystemExit("parent native evaluation is not frozen")
PY

mkdir -p "$threshold_dir/slurm"
shard_job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 02:00:00 \
  --array=0-2%3 -J fcup50_threshold \
  -o "$threshold_dir/slurm/shard_%A_%a.out" -e "$threshold_dir/slurm/shard_%A_%a.err" \
  --export="ALL,PHC2D_UPGRADE_50K_THRESHOLD_DIR=$threshold_dir" \
  jobs/paracloud_full_cell_upgrade_50k_threshold.sh)"
finalizer_job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 01:00:00 \
  --dependency="afterok:$shard_job_id" -J fcup50_threshold_finalize \
  -o "$threshold_dir/slurm/finalizer_%j.out" -e "$threshold_dir/slurm/finalizer_%j.err" \
  --export="ALL,PHC2D_UPGRADE_50K_THRESHOLD_DIR=$threshold_dir" \
  jobs/paracloud_full_cell_upgrade_50k_threshold_finalize.sh)"
{
  echo "authorization_id=$authorization_id"
  echo "deployed_git_commit=$actual_commit"
  echo "shard_job_id=$shard_job_id"
  echo "finalizer_job_id=$finalizer_job_id"
  echo "submitted_at=$(date -Is)"
} > "$threshold_dir/submission.env"
printf 'shard_job_id=%s\nfinalizer_job_id=%s\n' "$shard_job_id" "$finalizer_job_id"
