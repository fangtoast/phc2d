#!/usr/bin/env bash
# Submit a frozen Full-cell parameter-ablation manifest as a bounded GPU array.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail

: "${PHC2D_ROOT:?PHC2D_ROOT is required}"
: "${PHC2D_AUTHORIZE_PARAMETER_ABLATION:?Set PHC2D_AUTHORIZE_PARAMETER_ABLATION=YES after reviewing the manifest}"
if [[ "$PHC2D_AUTHORIZE_PARAMETER_ABLATION" != "YES" ]]; then
  printf '%s\n' "status=authorization_rejected"
  exit 2
fi

manifest="${1:?usage: $0 MANIFEST [CONCURRENCY [OFFSET [LIMIT]]]}"
concurrency="${2:-3}"
offset="${3:-0}"
if [[ ! -f "$manifest" ]]; then
  printf 'status=manifest_missing path=%s\n' "$manifest"
  exit 2
fi
if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]] || (( concurrency > 8 )); then
  printf 'status=invalid_concurrency value=%s allowed=1..8\n' "$concurrency"
  exit 2
fi
if [[ ! "$offset" =~ ^[0-9]+$ ]]; then
  printf 'status=invalid_offset value=%s\n' "$offset"
  exit 2
fi

python_bin="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}/bin/python"
case_count="$($python_bin -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], encoding="utf-8"))))' "$manifest")"
if (( case_count < 1 )); then
  printf '%s\n' "status=empty_manifest"
  exit 2
fi
if (( offset >= case_count )); then
  printf 'status=offset_out_of_range offset=%s cases=%s\n' "$offset" "$case_count"
  exit 2
fi
remaining="$((case_count - offset))"
limit="${4:-$remaining}"
if [[ ! "$limit" =~ ^[1-9][0-9]*$ ]] || (( limit > remaining || limit > 15 )); then
  printf 'status=invalid_limit value=%s remaining=%s qos_max_chunk=15\n' "$limit" "$remaining"
  exit 2
fi

job_id="$(sbatch --parsable \
  --job-name=phc2d-param-ablation \
  --partition=gpu_4090 \
  --gres=gpu:1 \
  --cpus-per-task=6 \
  --array="0-$((limit - 1))%${concurrency}" \
  --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_PARAMETER_MANIFEST=$(realpath "$manifest"),PHC2D_PARAMETER_INDEX_OFFSET=$offset,PHC2D_ENV=${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118},PHC2D_DEPLOYED_GIT_COMMIT=${PHC2D_DEPLOYED_GIT_COMMIT:-unknown}" \
  --output="$(dirname "$manifest")/slurm-%A_%a.out" \
  --error="$(dirname "$manifest")/slurm-%A_%a.err" \
  "$PHC2D_ROOT/jobs/paracloud_full_cell_parameter_ablation.sh")"
printf 'status=submitted job_id=%s total_cases=%s offset=%s submitted_cases=%s concurrency=%s manifest=%s\n' "$job_id" "$case_count" "$offset" "$limit" "$concurrency" "$manifest"
