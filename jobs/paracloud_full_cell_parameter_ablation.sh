#!/usr/bin/env bash
# Run one frozen Full-cell parameter-ablation case on one allocated GPU.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail

: "${PHC2D_ROOT:?PHC2D_ROOT is required}"
: "${PHC2D_PARAMETER_MANIFEST:?PHC2D_PARAMETER_MANIFEST is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
export PYTHONPATH="$PHC2D_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PHC2D_DEPLOYED_GIT_COMMIT="${PHC2D_DEPLOYED_GIT_COMMIT:-unknown}"
case_index="$((SLURM_ARRAY_TASK_ID + ${PHC2D_PARAMETER_INDEX_OFFSET:-0}))"

"$PHC2D_ENV/bin/python" -m phc2d.forward.full_cell_parameter_ablation run-case \
  --manifest "$PHC2D_PARAMETER_MANIFEST" \
  --index "$case_index"
