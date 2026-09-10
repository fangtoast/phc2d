#!/usr/bin/env bash
# Run the EXP-FWD-010 fair-total-data joint training task on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD010_RUN_DIR:?PHC2D_FWD010_RUN_DIR is required}"
PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_dual_lane \
  --config configs/research/full_cell_k5_forward_010_dual_lane_4k_seed0_v1.yaml \
  --seed 0 --head-kind dense --run-dir "$PHC2D_FWD010_RUN_DIR"
