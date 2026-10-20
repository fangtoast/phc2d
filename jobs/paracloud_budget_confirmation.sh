#!/usr/bin/env bash
# Execute isolated fixed-budget tasks in one GPU allocation.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment}"
: "${PHC2D_CAMPAIGN:?Set frozen campaign}"
: "${PHC2D_STAGE:?Set admission or train}"
: "${PHC2D_BUNDLE_SLOTS:?Set number of disjoint slots}"
: "${SLURM_ARRAY_TASK_ID:?Run as a Slurm array}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
python -m phc2d.forward.budget_confirmation "$PHC2D_STAGE" \
  --campaign "$PHC2D_CAMPAIGN" --slot "$SLURM_ARRAY_TASK_ID" --slots "$PHC2D_BUNDLE_SLOTS"
