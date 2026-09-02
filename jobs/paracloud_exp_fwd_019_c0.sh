#!/usr/bin/env bash
# Run one frozen EXP-FWD-019 C0 validation diagnostic on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD019_C0_MANIFEST:?Set PHC2D_FWD019_C0_MANIFEST.}"
: "${SLURM_ARRAY_TASK_ID:?Run as a Slurm array task.}"
row="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$PHC2D_FWD019_C0_MANIFEST")"
row="${row%$'\r'}"
IFS=, read -r task_index case_id model lane config checkpoint output_dir <<< "$row"
[[ "$task_index" == "$SLURM_ARRAY_TASK_ID" ]] || { echo "manifest identity mismatch" >&2; exit 2; }
python -m phc2d.forward.component_attribution \
  --config "$config" --checkpoint "$checkpoint" --output-dir "$output_dir"
