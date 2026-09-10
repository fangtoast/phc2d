#!/usr/bin/env bash
# Run one frozen Full-cell parameter-ablation case on one allocated GPU.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"

: "${PHC2D_PARAMETER_MANIFEST:?PHC2D_PARAMETER_MANIFEST is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

case_index="$((SLURM_ARRAY_TASK_ID + ${PHC2D_PARAMETER_INDEX_OFFSET:-0}))"

python -m phc2d.forward.full_cell_parameter_ablation run-case \
  --manifest "$PHC2D_PARAMETER_MANIFEST" \
  --index "$case_index"
