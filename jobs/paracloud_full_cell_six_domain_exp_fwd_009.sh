#!/usr/bin/env bash
# Run one EXP-FWD-009 six-domain evaluation shard on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD009_SIX_DOMAIN_CAMPAIGN:?campaign path required}"
: "${SLURM_ARRAY_TASK_ID:?array task id required}"
PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_six_domain_n40 run-shard \
  --campaign-dir "$PHC2D_FWD009_SIX_DOMAIN_CAMPAIGN" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --repository-root "$PHC2D_ROOT" \
  --device cuda:0
