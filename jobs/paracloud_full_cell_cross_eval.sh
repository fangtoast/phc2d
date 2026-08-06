#!/usr/bin/env bash
# Execute the frozen 576-cell full-cell cross-evaluation matrix on one GPU.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_CROSS_WAVES_FILE:?Set PHC2D_CROSS_WAVES_FILE.}"
: "${PHC2D_CROSS_OUTPUT_DIR:?Set PHC2D_CROSS_OUTPUT_DIR.}"

wave_args=()
while IFS=, read -r label manifest; do
  [[ "$label" == "label" ]] && continue
  wave_args+=(--wave "$label=$manifest")
done < "$PHC2D_CROSS_WAVES_FILE"

python -m phc2d.forward.full_cell_cross_eval \
  "${wave_args[@]}" \
  --binary-dataset data/full_cell_binary_bz9_10000_xingyi_v1.h5 \
  --graded-dataset data/full_cell_graded_bz9_10000_paracloud_m9_v1.h5 \
  --output-dir "$PHC2D_CROSS_OUTPUT_DIR" \
  --device cuda:0
