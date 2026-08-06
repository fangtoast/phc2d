#!/usr/bin/env bash
# Submit the two-lane Xingyi 50k campaign; dry-run unless explicitly authorized.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
PHC2D_RUN_ROOT="${PHC2D_RUN_ROOT:-$HOME/HDD_POOL/runs/$(date +%Y%m%d)}"
PHC2D_CAMPAIGN_ID="${PHC2D_CAMPAIGN_ID:-full_cell_dual_lane_50000_xingyi_$(date +%Y%m%d%H%M)}"
PHC2D_SUBMIT_MODE="${PHC2D_SUBMIT_MODE:-dry-run}"
PHC2D_PLAN_MODE="${PHC2D_PLAN_MODE:-frozen}"
campaign_root="$PHC2D_RUN_ROOT/$PHC2D_CAMPAIGN_ID"
binary_run="$campaign_root/full_cell_binary_bz9_50000_xingyi_v1"
graded_run="$campaign_root/full_cell_graded_bz9_50000_xingyi_v1"

for run_dir in "$binary_run" "$graded_run"; do
  if [[ "$PHC2D_PLAN_MODE" == "generate-on-node" ]]; then
    if [[ -e "$run_dir/results.json" || -d "$run_dir/worker_shards" ]]; then
      echo "generate-on-node requires a fresh run directory: $run_dir" >&2
      exit 2
    fi
  elif [[ "$PHC2D_PLAN_MODE" == "frozen" ]]; then
    if [[ ! -f "$run_dir/sample_plan.jsonl" ]]; then
      echo "Missing frozen plan: $run_dir/sample_plan.jsonl" >&2
      exit 2
    fi
    if [[ "$(wc -l < "$run_dir/sample_plan.jsonl")" -ne 50000 ]]; then
      echo "Plan must contain exactly 50000 rows: $run_dir/sample_plan.jsonl" >&2
      exit 2
    fi
  else
    echo "PHC2D_PLAN_MODE must be frozen or generate-on-node." >&2
    exit 2
  fi
done

submit_lane() {
  local lane="$1"
  local run_dir="$2"
  local log_path="$campaign_root/${lane}_%j.log"
  local args=(
    -p xyfree -N 1 -n 1 -c 64 --mem=128G --time=24:00:00
    --job-name="phc2d_50k_${lane}"
    --output="$log_path"
    --export="ALL,PHC2D_AUTHORIZE_50K=YES,PHC2D_LANE=$lane,PHC2D_RUN_DIR=$run_dir,PHC2D_WORKERS=64,PHC2D_ALLOW_PLAN_CREATION=$([[ $PHC2D_PLAN_MODE == generate-on-node ]] && printf 1 || printf 0)"
    "$PHC2D_ROOT/jobs/xingyi_full_cell_50000.sh"
  )
  if [[ "$PHC2D_SUBMIT_MODE" == "dry-run" ]]; then
    printf 'sbatch' >&2
    printf ' %q' "${args[@]}" >&2
    printf '\n' >&2
    printf 'DRYRUN-%s\n' "$lane"
    return
  fi
  if [[ "$PHC2D_SUBMIT_MODE" != "submit" || "${PHC2D_AUTHORIZE_50K:-NO}" != "YES" ]]; then
    echo "Submission requires PHC2D_SUBMIT_MODE=submit and PHC2D_AUTHORIZE_50K=YES." >&2
    exit 3
  fi
  sbatch --parsable "${args[@]}"
}

mkdir -p "$campaign_root"
binary_job="$(submit_lane binary "$binary_run")"
graded_job="$(submit_lane graded "$graded_run")"
printf 'lane,job_id,run_dir,mode\n' > "$campaign_root/submission_manifest.csv"
printf 'binary,%s,%s,%s\n' "$binary_job" "$binary_run" "$PHC2D_SUBMIT_MODE" >> "$campaign_root/submission_manifest.csv"
printf 'graded,%s,%s,%s\n' "$graded_job" "$graded_run" "$PHC2D_SUBMIT_MODE" >> "$campaign_root/submission_manifest.csv"
printf '%s\n' "$campaign_root/submission_manifest.csv"
