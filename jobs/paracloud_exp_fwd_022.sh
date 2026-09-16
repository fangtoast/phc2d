#!/usr/bin/env bash
# Run one EXP-FWD-022 frozen-attribution stage on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
: "${PHC2D_FWD022_CAMPAIGN:?Set PHC2D_FWD022_CAMPAIGN}"
: "${PHC2D_FWD022_STAGE:?Set PHC2D_FWD022_STAGE}"

cd "$PHC2D_ROOT"
export PYTHONPATH="$PHC2D_ROOT/src"
export PATH="$PHC2D_ENV/bin:$PATH"

case "$PHC2D_FWD022_STAGE" in
  preflight)
    "$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_branch_attribution preflight \
      --campaign-dir "$PHC2D_FWD022_CAMPAIGN" --repository-root "$PHC2D_ROOT"
    ;;
  shard)
    : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
    "$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_branch_attribution run-shard \
      --campaign-dir "$PHC2D_FWD022_CAMPAIGN" --shard-id "$SLURM_ARRAY_TASK_ID" \
      --repository-root "$PHC2D_ROOT"
    ;;
  finalize)
    "$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_branch_attribution finalize \
      --campaign-dir "$PHC2D_FWD022_CAMPAIGN" --repository-root "$PHC2D_ROOT"
    ;;
  *)
    echo "unsupported EXP-FWD-022 stage: $PHC2D_FWD022_STAGE" >&2
    exit 2
    ;;
esac
