#!/usr/bin/env bash
# Run one K/lane member of the paired Fourier-cutoff study on Xingyi.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
source "$PHC2D_ROOT/jobs/xingyi_common.sh"
: "${PHC2D_CUTOFF_ROOT:?Set the fixed campaign root.}"
: "${PHC2D_AUTHORIZE_FOURIER_CUTOFF:?Explicit authorization is required.}"
[[ "$PHC2D_AUTHORIZE_FOURIER_CUTOFF" == YES ]] || exit 3
task="${SLURM_ARRAY_TASK_ID:?Array task id is required}"
if (( task < 16 )); then lane=binary; cutoff=$((task + 5)); else lane=graded; cutoff=$((task - 16 + 5)); fi
config="configs/research/full_cell_${lane}_fourier_cutoff_pair1_xingyi_v1.yaml"
run_dir="$PHC2D_CUTOFF_ROOT/${lane}_k$(printf '%02d' "$cutoff")"
mkdir -p "$run_dir"
python -m phc2d.cli data generate --config "$config" --run-dir "$run_dir" \
  --fourier-cutoff "$cutoff" --dataset-output "$run_dir/dataset.h5" --workers 1 --plan-workers 1
