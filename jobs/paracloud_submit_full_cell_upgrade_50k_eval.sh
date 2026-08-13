#!/usr/bin/env bash
# Submit the authorized Full-Wh upgrade 50k evaluation shards and finalizer.
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
if [[ "${PHC2D_AUTHORIZE_UPGRADE_50K_EVAL:-}" != "YES" || "$authorization_id" != "${PHC2D_EXPECTED_50K_EVAL_AUTHORIZATION_ID:-}" ]]; then
  echo "upgrade 50k evaluation authorization mismatch" >&2
  exit 4
fi
for artifact in config.resolved.yaml evaluation_plan.csv checkpoint_inventory.csv campaign_status.json; do
  test -s "$campaign_dir/$artifact" || { echo "evaluation campaign missing $artifact" >&2; exit 5; }
done
if grep -q '^shard_job_id=' "$campaign_dir/submission.env" 2>/dev/null; then
  echo "evaluation campaign was already submitted" >&2
  exit 5
fi
python3 - "$campaign_dir" <<'PY'
import hashlib
import json
import pathlib
import sys

import yaml

campaign = pathlib.Path(sys.argv[1])
config = yaml.safe_load((campaign / "config.resolved.yaml").read_text(encoding="utf-8"))
status = json.loads((campaign / "campaign_status.json").read_text(encoding="utf-8"))
if status.get("status") != "evaluation_plan_ready" or status.get("test_metrics_read") is not False:
    raise SystemExit("evaluation plan is not frozen before test")
if hashlib.sha256((campaign / "evaluation_plan.csv").read_bytes()).hexdigest() != config["plan_hash"]:
    raise SystemExit("evaluation plan hash mismatch")
root = pathlib.Path.cwd()
for lane, contract in config["datasets"].items():
    path = root / contract["path"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != contract["sha256"]:
        raise SystemExit(f"dataset hash mismatch for lane {lane}")
for key, hash_key in (("training_campaign", "training_manifest_sha256"), ("baseline_inventory", "baseline_inventory_sha256")):
    path = root / config[key]
    if key == "training_campaign":
        path = path / "task_manifest.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != config[hash_key]:
        raise SystemExit(f"frozen input hash mismatch: {key}")
PY

mkdir -p "$campaign_dir/slurm"
shard_job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 04:00:00 \
  --array=0-7%8 -J fcup50_eval \
  -o "$campaign_dir/slurm/shard_%A_%a.out" -e "$campaign_dir/slurm/shard_%A_%a.err" \
  --export="ALL,PHC2D_UPGRADE_50K_EVAL_CAMPAIGN=$campaign_dir" \
  jobs/paracloud_full_cell_upgrade_50k_eval.sh)"
{
  echo "authorization_id=$authorization_id"
  echo "deployed_git_commit=$actual_commit"
  echo "shard_job_id=$shard_job_id"
  echo "submitted_at=$(date -Is)"
} > "$campaign_dir/submission.env"
finalizer_job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 02:00:00 \
  --dependency="afterok:$shard_job_id" -J fcup50_finalize \
  -o "$campaign_dir/slurm/finalizer_%j.out" -e "$campaign_dir/slurm/finalizer_%j.err" \
  --export="ALL,PHC2D_UPGRADE_50K_EVAL_CAMPAIGN=$campaign_dir" \
  jobs/paracloud_full_cell_upgrade_50k_eval_finalize.sh)"
{
  echo "finalizer_job_id=$finalizer_job_id"
  echo "finalizer_submitted_at=$(date -Is)"
} >> "$campaign_dir/submission.env"
printf 'shard_job_id=%s\nfinalizer_job_id=%s\n' "$shard_job_id" "$finalizer_job_id"
