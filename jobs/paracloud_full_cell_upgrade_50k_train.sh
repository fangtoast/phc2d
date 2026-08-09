#!/usr/bin/env bash
# Run one frozen Full-Wh upgrade 50k confirmation task on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"

: "${PHC2D_UPGRADE_50K_MANIFEST:?Set PHC2D_UPGRADE_50K_MANIFEST.}"
: "${SLURM_ARRAY_TASK_ID:?This worker must run as a Slurm array task.}"
row="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$PHC2D_UPGRADE_50K_MANIFEST")"
row="${row%$'\r'}"
if [[ -z "$row" ]]; then
  echo "missing 50k manifest row for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
IFS=, read -r task_index wave slot task_id action model head lane seed config run_dir <<< "$row"
if [[ "$task_index" != "$SLURM_ARRAY_TASK_ID" || "$action" != "RUN" ]]; then
  echo "50k manifest task identity/action mismatch" >&2
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
  python3 - "$profile" "$run_dir/n40_profile.json" <<'PY' || true
import csv
import json
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
payload = {
    "status": "profile_collected" if rows else "profile_empty",
    "sample_interval_s": 60,
    "sample_count": len(rows),
    "gpu_uuids": sorted({row["gpu_uuid"].strip() for row in rows if row.get("gpu_uuid")}),
    "utilization_gpu_pct_mean": sum(float(row["utilization_gpu_pct"]) for row in rows) / max(len(rows), 1),
    "utilization_gpu_pct_max": max((float(row["utilization_gpu_pct"]) for row in rows), default=0.0),
    "memory_used_mib_max": max((float(row["memory_used_mib"]) for row in rows), default=0.0),
    "power_w_max": max((float(row["power_w"]) for row in rows), default=0.0),
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY
}
trap cleanup_profile EXIT

args=(--config "$config" --seed "$seed" --head-kind "$head" --run-dir "$run_dir")
if [[ "$lane" == "J" ]]; then
  python -m phc2d.forward.full_cell_dual_lane "${args[@]}"
else
  python -m phc2d.forward.full_cell --model "$model" "${args[@]}"
fi
test -f "$run_dir/model.pt"
test -f "$run_dir/epoch100.pt"
test -f "$run_dir/model_summary.json"
test ! -e "$run_dir/checkpoint.pt"
ln "$run_dir/model.pt" "$run_dir/checkpoint.pt"
