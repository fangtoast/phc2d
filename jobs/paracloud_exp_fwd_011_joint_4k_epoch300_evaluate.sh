#!/usr/bin/env bash
# Run the six affected EXP-FWD-011 joint evaluation cells on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD011_EVAL_DIR:?PHC2D_FWD011_EVAL_DIR is required}"
PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_joint_fair_evaluation run \
  --campaign-dir "$PHC2D_FWD011_EVAL_DIR" --repository-root "$PHC2D_ROOT" --device cuda:0
