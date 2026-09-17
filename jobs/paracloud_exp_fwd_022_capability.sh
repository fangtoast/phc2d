#!/usr/bin/env bash
# Execute one frozen EXP-FWD-022 independent-capability stage on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
: "${PHC2D_ROOT:?Set PHC2D_ROOT}"
: "${PHC2D_FWD022_CAPABILITY:?Set PHC2D_FWD022_CAPABILITY}"
: "${PHC2D_FWD022_CAP_STAGE:?Set PHC2D_FWD022_CAP_STAGE}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"

args=("$PHC2D_FWD022_CAP_STAGE" --campaign-dir "$PHC2D_FWD022_CAPABILITY")
if [[ "$PHC2D_FWD022_CAP_STAGE" == "train" ]]; then
  : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
  args+=(--task-index "$SLURM_ARRAY_TASK_ID")
fi
python -m phc2d.forward.global_local_branch_capability "${args[@]}"
