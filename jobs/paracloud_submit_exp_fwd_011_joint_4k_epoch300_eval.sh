#!/usr/bin/env bash
# Freeze, dry-run, or explicitly submit the EXP-FWD-011 six-domain evaluation.
# Failed submissions are recorded; no automatic retry is allowed.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
mode="${1:-}"
authorization_id="${2:-}"
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
: "${PHC2D_FWD011_TRAIN_DIR:?PHC2D_FWD011_TRAIN_DIR is required}"
: "${PHC2D_FWD011_EVAL_DIR:?PHC2D_FWD011_EVAL_DIR is required}"
cd "$PHC2D_ROOT"
if [[ "$mode" == "freeze" ]]; then
  PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_joint_fair_evaluation freeze \
    --config configs/research/full_cell_k5_joint_4k_epoch300_six_domain_exp_fwd_011_n40_v1.yaml \
    --checkpoint "$PHC2D_FWD011_TRAIN_DIR/model.pt" --campaign-dir "$PHC2D_FWD011_EVAL_DIR" \
    --repository-root "$PHC2D_ROOT"
  exit 0
fi
[[ "$mode" == "dry-run" || "$mode" == "submit" ]] || { echo "usage: $0 freeze|dry-run|submit AUTHORIZATION_ID" >&2; exit 2; }
[[ -f "$PHC2D_FWD011_EVAL_DIR/evaluation_plan.csv" ]] || { echo "frozen evaluation required" >&2; exit 3; }
if [[ "$mode" == "dry-run" ]]; then
  echo "evaluation_dir=$PHC2D_FWD011_EVAL_DIR cells=6 sbatch -p gpu_4090 --gres=gpu:1 -c 6 -t 02:00:00"
  exit 0
fi
[[ "${PHC2D_AUTHORIZE_EXP_FWD_011:-}" == "YES" ]] || { echo "submit requires PHC2D_AUTHORIZE_EXP_FWD_011=YES" >&2; exit 4; }
[[ -n "$authorization_id" ]] || { echo "explicit authorization id is required" >&2; exit 4; }
set +e
submit_output="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 02:00:00 -J fwd011_eval \
  -o "$PHC2D_FWD011_EVAL_DIR/slurm_%j.out" -e "$PHC2D_FWD011_EVAL_DIR/slurm_%j.err" \
  --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_FWD011_EVAL_DIR=$PHC2D_FWD011_EVAL_DIR" \
  jobs/paracloud_exp_fwd_011_joint_4k_epoch300_evaluate.sh 2>&1)"
submit_status=$?
set -e
if [[ "$submit_status" -ne 0 ]]; then
  printf 'failed_at=%s\nsbatch_output=%s\nnext_action=human_diagnosis_and_explicit_resubmission\n' "$(date -Is)" "$submit_output" > "$PHC2D_FWD011_EVAL_DIR/submission_failure.txt"
  echo "sbatch failed; no automatic retry" >&2
  exit "$submit_status"
fi
printf 'authorization_id=%s\njob_id=%s\nsubmitted_at=%s\n' "$authorization_id" "$submit_output" "$(date -Is)" > "$PHC2D_FWD011_EVAL_DIR/submission.env"
echo "job_id=$submit_output evaluation_dir=$PHC2D_FWD011_EVAL_DIR"
