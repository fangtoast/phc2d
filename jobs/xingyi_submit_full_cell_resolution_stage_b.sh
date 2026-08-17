#!/usr/bin/env bash
# Submit the Xingyi 100-structure MPB resolution Stage-B confirmation.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
PHC2D_RUN_ROOT="${PHC2D_RUN_ROOT:-$HOME/HDD_POOL/runs/$(date +%Y%m%d)}"
PHC2D_CAMPAIGN_ID="${PHC2D_CAMPAIGN_ID:-full_cell_resolution_convergence_xingyi_stage_b_$(date +%Y%m%d%H%M)}"
PHC2D_SUBMIT_MODE="${PHC2D_SUBMIT_MODE:-dry-run}"
run_dir="$PHC2D_RUN_ROOT/$PHC2D_CAMPAIGN_ID"
args=(
  -p xyfree -N 1 -n 1 -c 64 --mem=128G --time=02:00:00
  --job-name=phc2d_mpb_res_b
  --output="$run_dir/slurm_%j.log"
  --export="ALL,PHC2D_AUTHORIZE_RESOLUTION_STAGE_B=YES,PHC2D_RUN_DIR=$run_dir"
  "$PHC2D_ROOT/jobs/xingyi_full_cell_resolution_stage_b.sh"
)

mkdir -p "$run_dir"
printf 'campaign_id=%s\nrun_dir=%s\nmode=%s\n' \
  "$PHC2D_CAMPAIGN_ID" "$run_dir" "$PHC2D_SUBMIT_MODE" > "$run_dir/submission.env"
if [[ "$PHC2D_SUBMIT_MODE" == "dry-run" ]]; then
  printf 'sbatch'
  printf ' %q' "${args[@]}"
  printf '\n'
  exit 0
fi
if [[ "$PHC2D_SUBMIT_MODE" != "submit" || "${PHC2D_AUTHORIZE_RESOLUTION_STAGE_B:-NO}" != "YES" ]]; then
  echo "Submission requires submit mode and explicit Stage-B authorization." >&2
  exit 3
fi
job_id="$(sbatch --parsable "${args[@]}")"
printf 'job_id=%s\n' "$job_id" >> "$run_dir/submission.env"
printf '%s\n' "$job_id"
