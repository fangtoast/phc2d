#!/usr/bin/env bash
# Dry-run-first submitter for Xingyi unified-Fourier 8+8 smoke or 64+64 pilot.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
PHC2D_RUN_ROOT="${PHC2D_RUN_ROOT:-$HOME/HDD_POOL/runs/$(date +%Y%m%d)}"
PHC2D_PHASE="${PHC2D_PHASE:-smoke}"
PHC2D_SUBMIT_MODE="${PHC2D_SUBMIT_MODE:-dry-run}"
campaign_id="full_cell_fourier_${PHC2D_PHASE}_xingyi_$(date +%Y%m%d%H%M)"
campaign_root="$PHC2D_RUN_ROOT/$campaign_id"

case "$PHC2D_PHASE" in
  smoke) cpus=8; memory=32G; time_limit=02:00:00; authorization=PHC2D_AUTHORIZE_FOURIER_SMOKE ;;
  pilot) cpus=64; memory=128G; time_limit=24:00:00; authorization=PHC2D_AUTHORIZE_FOURIER_PILOT ;;
  *) echo "PHC2D_PHASE must be smoke or pilot." >&2; exit 2 ;;
esac

submit_lane() {
  local lane="$1"
  local run_dir="$campaign_root/full_cell_${lane}_fourier_bz9_${PHC2D_PHASE}_v1"
  local log_path="$campaign_root/${lane}_%j.log"
  local export_value="ALL,PHC2D_PHASE=$PHC2D_PHASE,PHC2D_LANE=$lane,PHC2D_RUN_DIR=$run_dir,$authorization=YES"
  local args=(
    -p xyfree -N 1 -n 1 -c "$cpus" --mem="$memory" --time="$time_limit"
    --job-name="phc2d_fourier_${PHC2D_PHASE}_${lane}"
    --output="$log_path"
    --export="$export_value"
    "$PHC2D_ROOT/jobs/xingyi_full_cell_fourier_pilot.sh"
  )
  if [[ "$PHC2D_SUBMIT_MODE" == "dry-run" ]]; then
    printf 'sbatch' >&2
    printf ' %q' "${args[@]}" >&2
    printf '\n' >&2
    printf 'DRYRUN-%s\n' "$lane"
    return
  fi
  if [[ "$PHC2D_SUBMIT_MODE" != "submit" || "${!authorization:-NO}" != "YES" ]]; then
    echo "Submission requires mode=submit and explicit phase authorization." >&2
    exit 3
  fi
  sbatch --parsable "${args[@]}"
}

mkdir -p "$campaign_root"
binary_job="$(submit_lane binary)"
graded_job="$(submit_lane graded)"
printf 'lane,job_id,phase,mode\n' > "$campaign_root/submission_manifest.csv"
printf 'binary,%s,%s,%s\n' "$binary_job" "$PHC2D_PHASE" "$PHC2D_SUBMIT_MODE" >> "$campaign_root/submission_manifest.csv"
printf 'graded,%s,%s,%s\n' "$graded_job" "$PHC2D_PHASE" "$PHC2D_SUBMIT_MODE" >> "$campaign_root/submission_manifest.csv"
printf '%s\n' "$campaign_root/submission_manifest.csv"
