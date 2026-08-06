#!/usr/bin/env bash
# Train one full-cell inverse Stage-A variant/method/seed on one N40 GPU.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_GPU_PLATFORM="${PHC2D_GPU_PLATFORM:-n40}"
if [[ "$PHC2D_GPU_PLATFORM" != "n40" ]]; then
  echo "The frozen full-cell inverse Stage-A contract currently admits only N40." >&2
  exit 2
fi

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"

: "${PHC2D_VARIANT:?Set PHC2D_VARIANT to one frozen full-cell analytic variant.}"
: "${PHC2D_METHOD:?Set PHC2D_METHOD to one Stage-A trainable method.}"
: "${PHC2D_SEED:?Set PHC2D_SEED to 0, 1, or 2.}"
: "${PHC2D_RUN_DIR:?Set PHC2D_RUN_DIR to the stable standard runs path.}"
PHC2D_CONFIG="${PHC2D_CONFIG:-configs/research/full_cell_inverse_stage_a_train_n40_v1.yaml}"

case "$PHC2D_VARIANT" in
  binary_polar_holes_1|binary_polar_holes_2|binary_spectral_threshold|graded_rbf_fourier_1|graded_rbf_fourier_2) ;;
  *) echo "Unsupported PHC2D_VARIANT: $PHC2D_VARIANT" >&2; exit 2 ;;
esac
case "$PHC2D_METHOD" in
  direct_inverse_mlp|single_mode_generator|mixture_conditional_generator) ;;
  *) echo "Unsupported PHC2D_METHOD: $PHC2D_METHOD" >&2; exit 2 ;;
esac
case "$PHC2D_SEED" in
  0|1|2) ;;
  *) echo "PHC2D_SEED must be 0, 1, or 2." >&2; exit 2 ;;
esac

python -m phc2d.cli inverse full-cell-train \
  --config "$PHC2D_CONFIG" \
  --variant "$PHC2D_VARIANT" \
  --method "$PHC2D_METHOD" \
  --seed "$PHC2D_SEED" \
  --run-dir "$PHC2D_RUN_DIR"
