#!/usr/bin/env bash
# Run one fixed N40 cross-domain shard and collect GPU profile.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_CROSS_DOMAIN_CAMPAIGN:?Set PHC2D_CROSS_DOMAIN_CAMPAIGN.}"
: "${SLURM_ARRAY_TASK_ID:?This worker requires a Slurm array.}"
campaign="$PHC2D_CROSS_DOMAIN_CAMPAIGN"
profile_dir="$campaign/profiles/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID")"
mkdir -p "$profile_dir"
profile="$profile_dir/n40_profile.csv"
printf 'timestamp,gpu_uuid,gpu_name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_w,temperature_c\n' > "$profile"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits >> "$profile" 2>/dev/null || true
    sleep 10
  done
) &
profile_pid=$!
trap 'kill "$profile_pid" 2>/dev/null || true; wait "$profile_pid" 2>/dev/null || true' EXIT
PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_cross_domain_n40 run-shard --campaign-dir "$campaign" --shard-id "$SLURM_ARRAY_TASK_ID" --repository-root "$PHC2D_ROOT" --device cuda:0
