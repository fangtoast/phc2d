#!/usr/bin/env bash
# Run one K=20 native/cross-lane evaluation shard on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_K20_ROOT:-$HOME/run/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_K20_CAMPAIGN_DIR:?PHC2D_K20_CAMPAIGN_DIR is required}"
: "${SLURM_ARRAY_TASK_ID:?array task id is required}"
PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_k20_campaign run-eval-shard \
  --campaign-dir "$PHC2D_K20_CAMPAIGN_DIR" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --repository-root "$PHC2D_ROOT"
