#!/usr/bin/env bash
# Submit the authorized seed-0 native R2 supplement to one N40 GPU.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

mode="${1:-}"
campaign_dir="${2:-}"
authorization_id="${3:-}"
if [[ "$mode" != "submit" || -z "$campaign_dir" || -z "$authorization_id" ]]; then
  echo "usage: $0 submit CAMPAIGN_DIR AUTHORIZATION_ID" >&2
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
if [[ "${PHC2D_AUTHORIZE_UPGRADE_50K_R2:-}" != "YES" || "$authorization_id" != "${PHC2D_EXPECTED_50K_R2_AUTHORIZATION_ID:-}" ]]; then
  echo "upgrade 50k R2 authorization mismatch" >&2
  exit 4
fi
for artifact in config.resolved.yaml evaluation_plan.csv checkpoint_inventory.csv campaign_status.json; do
  test -s "$campaign_dir/$artifact" || { echo "R2 campaign missing $artifact" >&2; exit 5; }
done
if grep -q '^job_id=' "$campaign_dir/submission.env" 2>/dev/null; then
  echo "R2 campaign was already submitted" >&2
  exit 5
fi
python3 - "$campaign_dir" <<'PY'
import csv
import hashlib
import json
import pathlib
import sys

import yaml

campaign = pathlib.Path(sys.argv[1])
config = yaml.safe_load((campaign / "config.resolved.yaml").read_text(encoding="utf-8"))
status = json.loads((campaign / "campaign_status.json").read_text(encoding="utf-8"))
if status.get("status") != "r2_supplement_plan_ready" or status.get("test_metrics_read") is not False:
    raise SystemExit("R2 plan is not frozen before submission")
if config.get("seed") != 0 or config.get("expected_cells") != 18 or config.get("expected_checkpoints") != 17:
    raise SystemExit("R2 seed/cell/checkpoint contract mismatch")
if hashlib.sha256((campaign / "evaluation_plan.csv").read_bytes()).hexdigest() != config["plan_hash"]:
    raise SystemExit("R2 evaluation plan hash mismatch")
rows = list(csv.DictReader((campaign / "evaluation_plan.csv").open(encoding="utf-8")))
if len(rows) != 18 or {int(row["seed"]) for row in rows} != {0}:
    raise SystemExit("R2 plan is not exactly the 18 seed-0 cells")
root = pathlib.Path.cwd()
for lane, contract in config["datasets"].items():
    path = root / contract["path"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != contract["sha256"]:
        raise SystemExit(f"dataset hash mismatch for lane {lane}")
for row in rows:
    checkpoint = root / row["checkpoint_path"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != row["checkpoint_sha256"]:
        raise SystemExit(f"checkpoint hash mismatch: {checkpoint}")
PY

mkdir -p "$campaign_dir/slurm"
job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 --mem=60G -t 00:30:00 \
  -J fcup50_r2_s0 \
  -o "$campaign_dir/slurm/r2_%j.out" -e "$campaign_dir/slurm/r2_%j.err" \
  --export="ALL,PHC2D_UPGRADE_50K_R2_CAMPAIGN=$campaign_dir" \
  jobs/paracloud_full_cell_upgrade_50k_r2.sh)"
{
  echo "authorization_id=$authorization_id"
  echo "deployed_git_commit=$actual_commit"
  echo "job_id=$job_id"
  echo "submitted_at=$(date -Is)"
} > "$campaign_dir/submission.env"
printf 'job_id=%s\n' "$job_id"

