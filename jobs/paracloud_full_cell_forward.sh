#!/usr/bin/env bash
# Run one isolated full-cell forward model-plus-seed job on Paracloud GPU.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"

: "${PHC2D_FORWARD_CONFIG:?Set PHC2D_FORWARD_CONFIG.}"
: "${PHC2D_FORWARD_MODEL:?Set PHC2D_FORWARD_MODEL.}"
: "${PHC2D_FORWARD_SEED:?Set PHC2D_FORWARD_SEED.}"
: "${PHC2D_FORWARD_RUN_DIR:?Set PHC2D_FORWARD_RUN_DIR.}"

args=(
  --config "$PHC2D_FORWARD_CONFIG"
  --model "$PHC2D_FORWARD_MODEL"
  --seed "$PHC2D_FORWARD_SEED"
  --run-dir "$PHC2D_FORWARD_RUN_DIR"
)
if [[ "${PHC2D_FORWARD_ADMISSION_PROBE:-0}" == "1" ]]; then
  args+=(--admission-probe)
fi
if [[ -n "${PHC2D_FORWARD_THRESHOLD:-}" ]]; then
  args+=(--threshold "$PHC2D_FORWARD_THRESHOLD")
fi
python -m phc2d.forward.full_cell "${args[@]}"
