#!/usr/bin/env bash
# Plan or submit the K=20 34-cell native/cross-lane evaluation on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
mode="${1:-}"
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
PHC2D_PYTHON="${PHC2D_PYTHON:-$PHC2D_ENV/bin/python}"
PHC2D_CAMPAIGN_DIR="${PHC2D_CAMPAIGN_DIR:-}"
[[ -n "$PHC2D_CAMPAIGN_DIR" ]] || { echo "PHC2D_CAMPAIGN_DIR is required" >&2; exit 2; }
cd "$PHC2D_ROOT"

case "$mode" in
  plan)
    PYTHONPATH=src "$PHC2D_PYTHON" -m phc2d.forward.full_cell_k20_campaign plan-eval --campaign-dir "$PHC2D_CAMPAIGN_DIR"
    ;;
  dry-run)
    [[ -f "$PHC2D_CAMPAIGN_DIR/evaluation_cells_plan.csv" ]] || { echo "plan-eval is required" >&2; exit 3; }
    echo "campaign_dir=$PHC2D_CAMPAIGN_DIR"
    echo "sbatch -p gpu_4090 --gres=gpu:1 -c 6 -t 12:00:00 --array=0-7%8 -J fck20_eval"
    ;;
  submit)
    [[ "${PHC2D_AUTHORIZE_K20:-}" == "YES" ]] || { echo "submit requires PHC2D_AUTHORIZE_K20=YES" >&2; exit 4; }
    [[ -f "$PHC2D_CAMPAIGN_DIR/evaluation_cells_plan.csv" ]] || { echo "plan-eval is required" >&2; exit 3; }
    job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 12:00:00 --array=0-7%8 -J fck20_eval -o "$PHC2D_CAMPAIGN_DIR/slurm_eval_%A_%a.out" -e "$PHC2D_CAMPAIGN_DIR/slurm_eval_%A_%a.err" --export="ALL,PHC2D_K20_CAMPAIGN_DIR=$PHC2D_CAMPAIGN_DIR,PHC2D_K20_ROOT=$PHC2D_ROOT" jobs/paracloud_full_cell_k20_eval.sh)"
    printf 'wave=EVAL\njob_id=%s\nsubmitted_at=%s\n' "$job_id" "$(date -Is)" >> "$PHC2D_CAMPAIGN_DIR/submission.env"
    echo "wave=EVAL job_id=$job_id campaign_dir=$PHC2D_CAMPAIGN_DIR"
    ;;
  finalize)
    PYTHONPATH=src "$PHC2D_PYTHON" -m phc2d.forward.full_cell_k20_campaign finalize --campaign-dir "$PHC2D_CAMPAIGN_DIR" --repository-root "$PHC2D_ROOT"
    ;;
  *)
    echo "usage: $0 plan|dry-run|submit|finalize" >&2
    exit 2
    ;;
esac
