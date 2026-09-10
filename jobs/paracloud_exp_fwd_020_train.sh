#!/usr/bin/env bash
# Run one frozen EXP-FWD-020 G1 or G2 task on N56 RTX 5090.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d-exp-fwd-020}"
export PHC2D_GPU_PLATFORM=n56
export PHC2D_CUDA_MODULE=cuda/12.8
export PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD020_MANIFEST:?Set frozen task manifest.}"
: "${SLURM_ARRAY_TASK_ID:?Run as a Slurm array task.}"
row="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$PHC2D_FWD020_MANIFEST")"; row="${row%$'\r'}"
IFS=, read -r task_index case_id domain lane variant seed config run_dir <<< "$row"
[[ "$task_index" == "$SLURM_ARRAY_TASK_ID" ]] || { echo "manifest identity mismatch" >&2; exit 2; }
python -m phc2d.forward.full_cell --config "$config" --model global_local_band_operator_v1 --seed "$seed" --run-dir "$run_dir"
for artifact in run.log run.meta.json config.resolved.yaml metrics.json results.json history.csv model.pt model_summary.json; do test -s "$run_dir/$artifact"; done
