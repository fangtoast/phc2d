#!/usr/bin/env bash
# Run the full-cell single-card training preflight on N40 or N50.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"

: "${PHC2D_PREFLIGHT_RUN_DIR:?Set PHC2D_PREFLIGHT_RUN_DIR to a standard runs path.}"
PHC2D_PREFLIGHT_CONFIG="${PHC2D_PREFLIGHT_CONFIG:-configs/smoke/full_cell_gpu_single_card_preflight.yaml}"

python -m phc2d.training.gpu_cluster_preflight \
  --config "$PHC2D_PREFLIGHT_CONFIG" \
  --run-dir "$PHC2D_PREFLIGHT_RUN_DIR"
