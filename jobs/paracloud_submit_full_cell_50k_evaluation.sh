#!/usr/bin/env bash
# Submit the authorized eight-shard N40 evaluation campaign and its finalizer.
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
if [[ ! -f "$campaign_dir/evaluation_plan.csv" || ! -f "$campaign_dir/campaign_status.json" ]]; then
  echo "campaign plan is incomplete: $campaign_dir" >&2
  exit 2
fi
PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
if [[ ! -x "$PHC2D_ENV/bin/python" ]]; then
  echo "missing GPU Python environment: $PHC2D_ENV" >&2
  exit 2
fi
if [[ "$("$PHC2D_ENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$campaign_dir/campaign_status.json")" != "evaluation_plan_ready" ]]; then
  echo "campaign has already been submitted or is not ready" >&2
  exit 3
fi

mkdir -p "$campaign_dir/slurm"
shard_job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 00:30:00 \
  --array=0-7%8 -J fc50k_eval \
  -o "$campaign_dir/slurm/shard_%A_%a.out" -e "$campaign_dir/slurm/shard_%A_%a.err" \
  --export="ALL,PHC2D_EVALUATION_CAMPAIGN_DIR=$campaign_dir,PHC2D_EVALUATION_AUTHORIZATION_ID=$authorization_id" \
  jobs/paracloud_full_cell_50k_evaluation.sh)"
finalizer_job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 00:30:00 \
  --dependency="afterok:$shard_job_id" -J fc50k_eval_finalize \
  -o "$campaign_dir/slurm/finalizer_%j.out" -e "$campaign_dir/slurm/finalizer_%j.err" \
  --export="ALL,PHC2D_EVALUATION_CAMPAIGN_DIR=$campaign_dir,PHC2D_EVALUATION_AUTHORIZATION_ID=$authorization_id" \
  jobs/paracloud_full_cell_50k_evaluation_finalize.sh)"
PYTHONPATH="${PHC2D_ROOT:-$HOME/run/phc2d}/src" "$PHC2D_ENV/bin/python" -m phc2d.forward.full_cell_50k_evaluation record-submission \
  --campaign-dir "$campaign_dir" \
  --shard-job-id "$shard_job_id" \
  --finalizer-job-id "$finalizer_job_id"
printf 'shard_job_id=%s\nfinalizer_job_id=%s\n' "$shard_job_id" "$finalizer_job_id"
