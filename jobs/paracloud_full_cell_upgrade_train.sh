#!/usr/bin/env bash
# Run one manifest-selected upgraded Full-Wh training task on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"

: "${PHC2D_UPGRADE_MANIFEST:?Set PHC2D_UPGRADE_MANIFEST.}"
: "${SLURM_ARRAY_TASK_ID:?This worker must run as a Slurm array task.}"

row="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$PHC2D_UPGRADE_MANIFEST")"
if [[ -z "$row" ]]; then
  echo "missing manifest row for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
IFS=, read -r task_index task_id model head lane seed config run_dir admission <<< "$row"
if [[ "$task_index" != "$SLURM_ARRAY_TASK_ID" ]]; then
  echo "manifest task index mismatch: $task_index != $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

mkdir -p "$run_dir"
profile="$run_dir/n40_profile.csv"
printf 'timestamp,gpu_uuid,gpu_name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_w,temperature_c\n' > "$profile"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits >> "$profile" 2>/dev/null || true
    sleep 60
  done
) &
profile_pid=$!
cleanup_profile() {
  kill "$profile_pid" 2>/dev/null || true
  wait "$profile_pid" 2>/dev/null || true
  python - "$profile" "$run_dir/n40_profile.json" <<'PY' || true
import csv
import json
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
payload = {
    "status": "profile_collected" if rows else "profile_empty",
    "sample_interval_s": 60,
    "sample_count": len(rows),
    "gpu_uuids": sorted({row["gpu_uuid"].strip() for row in rows if row.get("gpu_uuid")}),
    "raw_csv": sys.argv[1],
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY
}
trap cleanup_profile EXIT

args=(--config "$config" --seed "$seed" --head-kind "$head" --run-dir "$run_dir")
if [[ "$admission" == "1" ]]; then
  args+=(--admission-probe)
fi
if [[ "$lane" == "J" ]]; then
  python -m phc2d.forward.full_cell_dual_lane "${args[@]}"
else
  python -m phc2d.forward.full_cell --model "$model" "${args[@]}"
fi
