#!/usr/bin/env bash
# Run the 15-case full-cell inverse Stage-A single-card preflight on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_GPU_PLATFORM="${PHC2D_GPU_PLATFORM:-n40}"
if [[ "$PHC2D_GPU_PLATFORM" != "n40" ]]; then
  echo "The frozen full-cell inverse Stage-A preflight currently admits only N40." >&2
  exit 2
fi

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"

: "${PHC2D_PREFLIGHT_RUN_DIR:?Set PHC2D_PREFLIGHT_RUN_DIR to a standard runs path.}"
PHC2D_PREFLIGHT_CONFIG="${PHC2D_PREFLIGHT_CONFIG:-configs/research/full_cell_inverse_stage_a_n40_preflight_v1.yaml}"
preflight_args=()
if [[ -n "${PHC2D_PREFLIGHT_VARIANT:-}" ]]; then
  preflight_args+=(--variant "$PHC2D_PREFLIGHT_VARIANT")
fi
if [[ -n "${PHC2D_PREFLIGHT_METHOD:-}" ]]; then
  preflight_args+=(--method "$PHC2D_PREFLIGHT_METHOD")
fi
if [[ -n "${PHC2D_PREFLIGHT_SEED:-}" ]]; then
  preflight_args+=(--seed "$PHC2D_PREFLIGHT_SEED")
fi

python -m phc2d.cli inverse full-cell-preflight \
  --config "$PHC2D_PREFLIGHT_CONFIG" \
  --run-dir "$PHC2D_PREFLIGHT_RUN_DIR" \
  "${preflight_args[@]}"
