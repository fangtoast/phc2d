#!/usr/bin/env bash
# Run one frozen K=20 seed-0 admission or training task on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_K20_ROOT:-$HOME/run/phc2d}"
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_K20_MANIFEST:?PHC2D_K20_MANIFEST is required}"
: "${SLURM_ARRAY_TASK_ID:?array task id is required}"
row="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$PHC2D_K20_MANIFEST")"
[[ -n "$row" ]] || { echo "missing manifest row" >&2; exit 2; }
IFS=, read -r task_index task_id model train_lane seed head grid_field config config_sha256 dataset_sha256 idx_train_hash idx_val_hash idx_test_hash k_points_sha256 run_dir gpu_uuid deployed_commit stage <<< "$row"
[[ "$task_index" == "$SLURM_ARRAY_TASK_ID" ]] || { echo "task index mismatch" >&2; exit 2; }
[[ "$stage" == "admission" || "$stage" == "formal" ]] || { echo "invalid manifest stage: $stage" >&2; exit 2; }
echo "task=$task_id stage=$stage manifest=$PHC2D_K20_MANIFEST"
mkdir -p "$run_dir"
profile="$run_dir/n40_profile.csv"
printf 'timestamp,gpu_uuid,gpu_name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_w,temperature_c\n' > "$profile"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits >> "$profile" 2>/dev/null || true
    sleep 60
  done
) &
profile_pid=$!
cleanup() { kill "$profile_pid" 2>/dev/null || true; wait "$profile_pid" 2>/dev/null || true; }
trap cleanup EXIT
args=(--manifest "$PHC2D_K20_MANIFEST" --task-index "$SLURM_ARRAY_TASK_ID" --stage "$stage" --repository-root "$PHC2D_ROOT")
[[ "$stage" == "admission" ]] && args+=(--admission-probe)
PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_k20_campaign _run-one "${args[@]}"
