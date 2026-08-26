#!/usr/bin/env bash
# Continue the EXP-FWD-010 fair joint checkpoint to total epoch 300 for EXP-FWD-011.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD011_SOURCE_RUN:?PHC2D_FWD011_SOURCE_RUN is required}"
: "${PHC2D_FWD011_RUN_DIR:?PHC2D_FWD011_RUN_DIR is required}"
PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_dual_lane \
  --config configs/research/full_cell_k5_forward_011_dual_lane_4k_epoch300_seed0_v1.yaml \
  --seed 0 --head-kind dense --warm-start-run "$PHC2D_FWD011_SOURCE_RUN" \
  --run-dir "$PHC2D_FWD011_RUN_DIR"
