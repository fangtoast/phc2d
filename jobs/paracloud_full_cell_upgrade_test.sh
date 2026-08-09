#!/usr/bin/env bash
# Evaluate one validation-frozen Full-Wh checkpoint on the native held-out test split.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"

: "${PHC2D_UPGRADE_TEST_MANIFEST:?Set PHC2D_UPGRADE_TEST_MANIFEST.}"
: "${SLURM_ARRAY_TASK_ID:?This worker must run as a Slurm array task.}"

row="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$PHC2D_UPGRADE_TEST_MANIFEST")"
row="${row%$'\r'}"
if [[ -z "$row" ]]; then
  echo "missing test manifest row for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
IFS=, read -r task_index task_id source_kind model head lane seed source_run_dir checkpoint_path checkpoint_sha256 result_dir <<< "$row"
if [[ "$task_index" != "$SLURM_ARRAY_TASK_ID" ]]; then
  echo "test manifest task index mismatch: $task_index != $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

campaign_dir="$(dirname "$PHC2D_UPGRADE_TEST_MANIFEST")"
profile_stage="$campaign_dir/profiles/$task_id"
mkdir -p "$profile_stage"
profile="$profile_stage/n40_profile.csv"
printf 'timestamp,gpu_uuid,gpu_name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_w,temperature_c\n' > "$profile"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits >> "$profile" 2>/dev/null || true
    sleep 10
  done
) &
profile_pid=$!
cleanup_profile() {
  kill "$profile_pid" 2>/dev/null || true
  wait "$profile_pid" 2>/dev/null || true
  python3 - "$profile" "$profile_stage/n40_profile.json" <<'PY' || true
import csv
import json
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
payload = {
    "status": "profile_collected" if rows else "profile_empty",
    "sample_interval_s": 10,
    "sample_count": len(rows),
    "gpu_uuids": sorted({row["gpu_uuid"].strip() for row in rows if row.get("gpu_uuid")}),
    "utilization_gpu_pct_max": max((float(row["utilization_gpu_pct"]) for row in rows), default=0.0),
    "memory_used_mib_max": max((float(row["memory_used_mib"]) for row in rows), default=0.0),
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY
  if [[ -d "$result_dir" ]]; then
    cp "$profile_stage/n40_profile.csv" "$result_dir/n40_profile.csv"
    cp "$profile_stage/n40_profile.json" "$result_dir/n40_profile.json"
  fi
}
trap cleanup_profile EXIT

python -m phc2d.forward.full_cell_upgrade_test run-task \
  --manifest "$PHC2D_UPGRADE_TEST_MANIFEST" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --repository-root "$PHC2D_ROOT"
