#!/usr/bin/env bash
# Finalize the frozen Full-Wh upgrade 50k evaluation campaign on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_UPGRADE_50K_EVAL_CAMPAIGN:?Set PHC2D_UPGRADE_50K_EVAL_CAMPAIGN.}"

python -m phc2d.forward.full_cell_upgrade_50k_evaluation finalize \
  --campaign-dir "$PHC2D_UPGRADE_50K_EVAL_CAMPAIGN" \
  --repository-root "$PHC2D_ROOT"
