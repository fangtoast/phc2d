#!/usr/bin/env bash
# Run one frozen EXP-FWD-020 G3 preflight or evaluation shard on N56.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d-exp-fwd-020}"
export PHC2D_GPU_PLATFORM=n56
export PHC2D_CUDA_MODULE=cuda/12.8
export PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_G3_CAMPAIGN:?Set frozen G3 campaign directory.}"
: "${PHC2D_G3_STAGE:?Set preflight or shard stage.}"

if [[ "$PHC2D_G3_STAGE" == preflight ]]; then
  python -m phc2d.forward.global_local_bridge_campaign preflight-g3 \
    --campaign-dir "$PHC2D_G3_CAMPAIGN" --repository-root "$PHC2D_ROOT" --device cuda:0
elif [[ "$PHC2D_G3_STAGE" == shard ]]; then
  : "${SLURM_ARRAY_TASK_ID:?G3 shard stage requires Slurm array task id.}"
  python -m phc2d.forward.global_local_bridge_campaign run-g3-shard \
    --campaign-dir "$PHC2D_G3_CAMPAIGN" --repository-root "$PHC2D_ROOT" \
    --shard-id "$SLURM_ARRAY_TASK_ID" --device cuda:0
  test -s "$PHC2D_G3_CAMPAIGN/shards/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID")/evaluation_cells.csv"
else
  echo "invalid G3 stage: $PHC2D_G3_STAGE" >&2
  exit 2
fi
