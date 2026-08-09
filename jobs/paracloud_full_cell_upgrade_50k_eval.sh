#!/usr/bin/env bash
# Run one frozen Full-Wh upgrade 50k evaluation shard on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_UPGRADE_50K_EVAL_CAMPAIGN:?Set PHC2D_UPGRADE_50K_EVAL_CAMPAIGN.}"
: "${SLURM_ARRAY_TASK_ID:?This worker requires a Slurm array task.}"

campaign_dir="$PHC2D_UPGRADE_50K_EVAL_CAMPAIGN"
profile_stage="$campaign_dir/profiles/shard_${SLURM_ARRAY_TASK_ID}"
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
    "utilization_gpu_pct_mean": sum(float(row["utilization_gpu_pct"]) for row in rows) / max(len(rows), 1),
    "utilization_gpu_pct_max": max((float(row["utilization_gpu_pct"]) for row in rows), default=0.0),
    "memory_used_mib_max": max((float(row["memory_used_mib"]) for row in rows), default=0.0),
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY
  shard_dir="$campaign_dir/shards/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID")"
  if [[ -d "$shard_dir" ]]; then
    cp "$profile_stage/n40_profile.csv" "$shard_dir/n40_profile.csv"
    cp "$profile_stage/n40_profile.json" "$shard_dir/n40_profile.json"
  fi
}
trap cleanup_profile EXIT

python -m phc2d.forward.full_cell_upgrade_50k_evaluation run-shard \
  --campaign-dir "$campaign_dir" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --repository-root "$PHC2D_ROOT"
