#!/usr/bin/env bash
# Dry-run or explicitly submit the single EXP-FWD-010 joint 4k+4k training task.
# Failed submissions are recorded; no automatic retry is allowed.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
mode="${1:-}"
authorization_id="${2:-}"
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
campaign="${PHC2D_FWD010_CAMPAIGN_DIR:-$PHC2D_ROOT/runs/$(date +%Y%m%d)/EXP-FWD-010_joint_4k_$(date +%Y%m%d%H%M%S)}"
run_dir="$campaign/train_joint_seed0"
cd "$PHC2D_ROOT"
[[ "$mode" == "dry-run" || "$mode" == "submit" ]] || { echo "usage: $0 dry-run|submit AUTHORIZATION_ID" >&2; exit 2; }
if [[ "$mode" == "dry-run" ]]; then
  echo "campaign_dir=$campaign sbatch -p gpu_4090 --gres=gpu:1 -c 6 -t 12:00:00"
  exit 0
fi
[[ "${PHC2D_AUTHORIZE_EXP_FWD_010:-}" == "YES" ]] || { echo "submit requires PHC2D_AUTHORIZE_EXP_FWD_010=YES" >&2; exit 4; }
[[ -n "$authorization_id" ]] || { echo "explicit authorization id is required" >&2; exit 4; }
mkdir -p "$campaign"
cp configs/research/full_cell_k5_forward_010_dual_lane_4k_seed0_v1.yaml "$campaign/config.frozen.yaml"
sha256sum "$campaign/config.frozen.yaml" > "$campaign/config.frozen.sha256"
set +e
submit_output="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 12:00:00 -J fwd010_joint4k \
  -o "$campaign/slurm_%j.out" -e "$campaign/slurm_%j.err" \
  --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_FWD010_RUN_DIR=$run_dir" \
  jobs/paracloud_exp_fwd_010_joint_4k_train.sh 2>&1)"
submit_status=$?
set -e
if [[ "$submit_status" -ne 0 ]]; then
  printf 'failed_at=%s\nsbatch_output=%s\nnext_action=human_diagnosis_and_explicit_resubmission\n' "$(date -Is)" "$submit_output" > "$campaign/submission_failure.txt"
  echo "sbatch failed; no automatic retry" >&2
  exit "$submit_status"
fi
printf 'authorization_id=%s\njob_id=%s\nrun_dir=%s\nsubmitted_at=%s\n' "$authorization_id" "$submit_output" "$run_dir" "$(date -Is)" > "$campaign/submission.env"
echo "job_id=$submit_output campaign_dir=$campaign"
