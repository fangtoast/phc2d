#!/usr/bin/env bash
# Run the single EXP-FWD-016 ranked-gap short-screen task on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD016_CAMPAIGN_DIR:?PHC2D_FWD016_CAMPAIGN_DIR is required}"
PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_dual_lane \
  --config "$PHC2D_FWD016_CAMPAIGN_DIR/config_ranked_v2.frozen.yaml" \
  --seed 0 \
  --head-kind dense \
  --run-dir "$PHC2D_FWD016_CAMPAIGN_DIR/train_ranked_v2_seed0"
