#!/usr/bin/env bash
# Freeze and submit six paired EXP-FWD-018 N40 training tasks.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
mode="${1:-}"
authorization_id="${2:-}"
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
campaign="${PHC2D_FWD018_CAMPAIGN_DIR:-$PHC2D_ROOT/runs/$(date +%Y%m%d)/EXP-FWD-018_frequency_weighted_$(date +%Y%m%d%H%M%S)}"
cd "$PHC2D_ROOT"
[[ "$mode" == "dry-run" || "$mode" == "submit" ]] || { echo "usage: $0 dry-run|submit AUTHORIZATION_ID" >&2; exit 2; }
if [[ "$mode" == "dry-run" ]]; then
  echo "campaign_dir=$campaign sbatch --array=0-5%3 -p gpu_4090 --gres=gpu:1 -c 6 -t 12:00:00"
  exit 0
fi
[[ "${PHC2D_AUTHORIZE_EXP_FWD_018:-}" == "YES" ]] || { echo "submit requires PHC2D_AUTHORIZE_EXP_FWD_018=YES" >&2; exit 4; }
[[ -n "$authorization_id" ]] || { echo "explicit authorization id is required" >&2; exit 4; }
[[ -f DEPLOYED_GIT_COMMIT ]] || { echo "missing DEPLOYED_GIT_COMMIT" >&2; exit 2; }
mkdir -p "$campaign"
cp configs/research/full_cell_k5_frequency_weighted_exp_fwd_018_baseline_v1.yaml "$campaign/config_baseline.frozen.yaml"
cp configs/research/full_cell_k5_frequency_weighted_exp_fwd_018_weighted_v1.yaml "$campaign/config_weighted.frozen.yaml"
sha256sum "$campaign"/config_*.frozen.yaml > "$campaign/configs.frozen.sha256"
printf 'authorization_id=%s\ndeployed_git_commit=%s\nsubmission_scope=validation_only_no_test\n' "$authorization_id" "$(tr -d '[:space:]' < DEPLOYED_GIT_COMMIT)" > "$campaign/submission.env"
set +e
submit_output="$(sbatch --parsable --array=0-5%3 -p gpu_4090 --gres=gpu:1 -c 6 -t 12:00:00 -J fwd018_freq \
  -o "$campaign/slurm_%A_%a.out" -e "$campaign/slurm_%A_%a.err" \
  --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_FWD018_CAMPAIGN_DIR=$campaign" \
  jobs/paracloud_exp_fwd_018_frequency_weighted_train.sh 2>&1)"
submit_status=$?
set -e
if [[ "$submit_status" -ne 0 ]]; then
  printf 'failed_at=%s\nsbatch_output=%s\nnext_action=human_diagnosis_and_explicit_resubmission\n' "$(date -Is)" "$submit_output" > "$campaign/submission_failure.txt"
  echo "sbatch failed; no automatic retry" >&2
  exit "$submit_status"
fi
printf 'job_id=%s\ncampaign_dir=%s\nsubmitted_at=%s\n' "$submit_output" "$campaign" "$(date -Is)" >> "$campaign/submission.env"
echo "job_id=$submit_output campaign_dir=$campaign"
