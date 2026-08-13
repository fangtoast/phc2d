#!/usr/bin/env bash
# Evaluate the three validation-frozen final full-cell configurations on test.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FINAL_SELECTION:?Set PHC2D_FINAL_SELECTION.}"
: "${PHC2D_MAIN_MANIFEST:?Set PHC2D_MAIN_MANIFEST.}"
: "${PHC2D_ABLATION_MANIFEST:?Set PHC2D_ABLATION_MANIFEST.}"
: "${PHC2D_FINAL_TEST_OUTPUT_DIR:?Set PHC2D_FINAL_TEST_OUTPUT_DIR.}"

python -m phc2d.forward.full_cell_final_test \
  --final-selection "$PHC2D_FINAL_SELECTION" \
  --main-matrix-manifest "$PHC2D_MAIN_MANIFEST" \
  --ablation-manifest "$PHC2D_ABLATION_MANIFEST" \
  --binary-dataset data/full_cell_binary_bz9_10000_xingyi_v1.h5 \
  --graded-dataset data/full_cell_graded_bz9_10000_paracloud_m9_v1.h5 \
  --output-dir "$PHC2D_FINAL_TEST_OUTPUT_DIR" \
  --device cuda:0
