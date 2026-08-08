#!/usr/bin/env bash
# Finalize the frozen 50k full-cell evaluation campaign after all GPU shards.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_EVALUATION_CAMPAIGN_DIR:?Set PHC2D_EVALUATION_CAMPAIGN_DIR.}"

python -m phc2d.forward.full_cell_50k_evaluation finalize \
  --campaign-dir "$PHC2D_EVALUATION_CAMPAIGN_DIR"
