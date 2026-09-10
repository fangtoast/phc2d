#!/usr/bin/env bash
# Run one frozen EXP-FWD-007 K5 admission or formal training task on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD007_MANIFEST:?PHC2D_FWD007_MANIFEST is required}"
: "${SLURM_ARRAY_TASK_ID:?array task id is required}"
stage="${PHC2D_FWD007_STAGE:?PHC2D_FWD007_STAGE is required}"
PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_training_campaign run-task \
  --manifest "$PHC2D_FWD007_MANIFEST" --task-index "$SLURM_ARRAY_TASK_ID" \
  --stage "$stage" --repository-root "$PHC2D_ROOT"
