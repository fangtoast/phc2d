#!/usr/bin/env bash
# Finalize the frozen Full-Wh winner threshold diagnostic on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_UPGRADE_50K_THRESHOLD_DIR:?Set PHC2D_UPGRADE_50K_THRESHOLD_DIR.}"

python -m phc2d.forward.full_cell_upgrade_50k_evaluation finalize-threshold \
  --threshold-dir "$PHC2D_UPGRADE_50K_THRESHOLD_DIR" \
  --repository-root "$PHC2D_ROOT"
