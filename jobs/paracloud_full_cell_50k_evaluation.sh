#!/usr/bin/env bash
# Run one frozen 50k full-cell evaluation shard on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_EVALUATION_CAMPAIGN_DIR:?Set PHC2D_EVALUATION_CAMPAIGN_DIR.}"
: "${SLURM_ARRAY_TASK_ID:?This job must be submitted as an array.}"

python -m phc2d.forward.full_cell_50k_evaluation run-shard \
  --campaign-dir "$PHC2D_EVALUATION_CAMPAIGN_DIR" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --device cuda:0
