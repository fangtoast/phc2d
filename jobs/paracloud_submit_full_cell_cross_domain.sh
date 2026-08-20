#!/usr/bin/env bash
# Freeze, preflight, dry-run, submit, or finalize the N40 four-domain campaign.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
mode="${1:-}"
campaign_dir="${2:-}"
authorization_id="${3:-N40_CROSS_DOMAIN_20260820_USER_AUTHORIZED}"
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
PHC2D_PYTHON="${PHC2D_PYTHON:-$PHC2D_ENV/bin/python}"
[[ -n "$campaign_dir" ]] || { echo "campaign directory is required" >&2; exit 2; }
cd "$PHC2D_ROOT"

case "$mode" in
  freeze)
    PYTHONPATH=src "$PHC2D_PYTHON" -m phc2d.forward.full_cell_cross_domain_n40 freeze --campaign-dir "$campaign_dir" --repository-root "$PHC2D_ROOT" ;;
  preflight)
    mkdir -p "$campaign_dir/slurm"
    preflight_job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 00:30:00 -J fc_cross_preflight -o "$campaign_dir/slurm/preflight_%j.out" -e "$campaign_dir/slurm/preflight_%j.err" --export="ALL,PHC2D_CROSS_DOMAIN_CAMPAIGN=$campaign_dir" jobs/paracloud_full_cell_cross_domain_preflight.sh)"
    echo "preflight_job_id=$preflight_job_id" > "$campaign_dir/preflight.env"
    echo "preflight_job_id=$preflight_job_id" ;;
  dry-run)
    test -s "$campaign_dir/evaluation_plan.csv"
    test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$campaign_dir/campaign_status.json")" = evaluation_plan_ready
    echo "sbatch -p gpu_4090 --gres=gpu:1 -c 6 -t 04:00:00 --array=0-7%8" ;;
  submit)
    [[ "${PHC2D_AUTHORIZE_N40_CROSS_DOMAIN:-}" == "YES" ]] || { echo "submit requires PHC2D_AUTHORIZE_N40_CROSS_DOMAIN=YES" >&2; exit 4; }
    [[ -f "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT" ]] || { echo "missing deployed commit marker" >&2; exit 5; }
    test -s "$campaign_dir/evaluation_plan.csv"
    test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$campaign_dir/campaign_status.json")" = evaluation_plan_ready
    expected_plan_hash="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$campaign_dir/evaluation_plan.csv")"
    actual_plan_hash="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_hash"])' "$campaign_dir/campaign_status.json")"
    [[ "$expected_plan_hash" == "$actual_plan_hash" ]] || { echo "plan hash mismatch" >&2; exit 6; }
    deployed_commit="$(tr -d '[:space:]' < "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT")"
    plan_commit="$(python3 -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["deployed_git_commit"])' "$campaign_dir/config.resolved.yaml")"
    [[ "$deployed_commit" == "$plan_commit" ]] || { echo "deployment commit mismatch" >&2; exit 6; }
    [[ ! -e "$campaign_dir/submission.env" ]] || { echo "campaign was already submitted" >&2; exit 7; }
    mkdir -p "$campaign_dir/slurm"
    shard_job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 04:00:00 --array=0-7%8 -J fc_cross_domain -o "$campaign_dir/slurm/shard_%A_%a.out" -e "$campaign_dir/slurm/shard_%A_%a.err" --export="ALL,PHC2D_CROSS_DOMAIN_CAMPAIGN=$campaign_dir" jobs/paracloud_full_cell_cross_domain.sh)"
    {
      echo "authorization_id=$authorization_id"
      echo "deployed_git_commit=$(tr -d '[:space:]' < "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT")"
      echo "shard_job_id=$shard_job_id"
      echo "submitted_at=$(date -Is)"
    } > "$campaign_dir/submission.env"
    finalizer_job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 02:00:00 --dependency="afterok:$shard_job_id" -J fc_cross_finalize -o "$campaign_dir/slurm/finalizer_%j.out" -e "$campaign_dir/slurm/finalizer_%j.err" --export="ALL,PHC2D_CROSS_DOMAIN_CAMPAIGN=$campaign_dir" jobs/paracloud_full_cell_cross_domain_finalize.sh)"
    echo "finalizer_job_id=$finalizer_job_id" >> "$campaign_dir/submission.env"
    PYTHONPATH=src "$PHC2D_PYTHON" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); s=json.loads((p/"campaign_status.json").read_text()); s.update({"status":"evaluation_submitted","shard_job_id":sys.argv[2],"finalizer_job_id":sys.argv[3],"next_gate":"8_shards_and_afterok_finalizer"}); (p/"campaign_status.json").write_text(json.dumps(s,indent=2)+"\n")' "$campaign_dir" "$shard_job_id" "$finalizer_job_id"
    printf 'shard_job_id=%s\nfinalizer_job_id=%s\n' "$shard_job_id" "$finalizer_job_id" ;;
  status)
    PYTHONPATH=src "$PHC2D_PYTHON" -m phc2d.forward.full_cell_cross_domain_n40 status --campaign-dir "$campaign_dir" --repository-root "$PHC2D_ROOT"
    command -v squeue >/dev/null 2>&1 && squeue -u "$USER" -o '%.18i %.12P %.24j %.10T %.12M %.12l %.8D %R' || true
    if command -v sacct >/dev/null 2>&1 && [[ -f "$campaign_dir/submission.env" ]]; then . "$campaign_dir/submission.env"; sacct -j "$shard_job_id,$finalizer_job_id" --format=JobID,State,ExitCode,Elapsed,MaxRSS -P || true; fi ;;
  finalize)
    PYTHONPATH=src "$PHC2D_PYTHON" -m phc2d.forward.full_cell_cross_domain_n40 finalize --campaign-dir "$campaign_dir" --repository-root "$PHC2D_ROOT" ;;
  *) echo "usage: $0 freeze|preflight|dry-run|submit|status|finalize CAMPAIGN_DIR [AUTHORIZATION_ID]" >&2; exit 2 ;;
esac
