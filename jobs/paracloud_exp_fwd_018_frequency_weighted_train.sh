#!/usr/bin/env bash
# Run one frozen EXP-FWD-018 loss arm and seed on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD018_CAMPAIGN_DIR:?PHC2D_FWD018_CAMPAIGN_DIR is required}"
case "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" in
  0) arm="baseline"; seed=0 ;;
  1) arm="weighted"; seed=0 ;;
  2) arm="baseline"; seed=1 ;;
  3) arm="weighted"; seed=1 ;;
  4) arm="baseline"; seed=2 ;;
  5) arm="weighted"; seed=2 ;;
  *) echo "unexpected EXP-FWD-018 task index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_dual_lane \
  --config "$PHC2D_FWD018_CAMPAIGN_DIR/config_${arm}.frozen.yaml" \
  --seed "$seed" --head-kind dense \
  --run-dir "$PHC2D_FWD018_CAMPAIGN_DIR/train_${arm}_seed${seed}"
