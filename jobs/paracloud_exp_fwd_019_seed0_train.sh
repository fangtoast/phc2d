#!/usr/bin/env bash
# Train one frozen EXP-FWD-019 seed0 component-screen task on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD019_SCREEN_MANIFEST:?Set PHC2D_FWD019_SCREEN_MANIFEST.}"
: "${SLURM_ARRAY_TASK_ID:?Run as a Slurm array task.}"
row="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$PHC2D_FWD019_SCREEN_MANIFEST")"; row="${row%$'\r'}"
IFS=, read -r task_index case_id family lane variant model seed config run_dir <<< "$row"
[[ "$task_index" == "$SLURM_ARRAY_TASK_ID" ]] || { echo "manifest identity mismatch" >&2; exit 2; }
if [[ "$lane" == "joint" ]]; then
  python -m phc2d.forward.full_cell_dual_lane --config "$config" --seed "$seed" --head-kind dense --run-dir "$run_dir"
else
  python -m phc2d.forward.full_cell --config "$config" --model "$model" --seed "$seed" --run-dir "$run_dir"
fi
test -s "$run_dir/results.json"; test -s "$run_dir/metrics.json"; test -s "$run_dir/model.pt"
