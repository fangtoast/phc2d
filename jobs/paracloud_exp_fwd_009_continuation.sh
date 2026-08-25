#!/usr/bin/env bash
# Run one EXP-FWD-009 preflight or weight-only continuation task.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
: "${PHC2D_FWD009_MANIFEST:?PHC2D_FWD009_MANIFEST is required}"
: "${PHC2D_FWD009_MODE:?PHC2D_FWD009_MODE is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
python -m phc2d.forward.full_cell_continuation_campaign "${PHC2D_FWD009_MODE}-task" \
  --manifest "$PHC2D_FWD009_MANIFEST" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --repository-root "$PHC2D_ROOT"
