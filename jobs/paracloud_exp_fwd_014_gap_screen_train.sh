#!/usr/bin/env bash
# Run one EXP-FWD-014 paired short-budget arm on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD014_CAMPAIGN_DIR:?PHC2D_FWD014_CAMPAIGN_DIR is required}"
case "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" in
  0) arm="baseline" ;;
  1) arm="gap_aware" ;;
  *) echo "unexpected array index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
config="$PHC2D_FWD014_CAMPAIGN_DIR/config_${arm}.frozen.yaml"
run_dir="$PHC2D_FWD014_CAMPAIGN_DIR/train_${arm}_seed0"
PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_dual_lane \
  --config "$config" --seed 0 --head-kind dense --run-dir "$run_dir"
