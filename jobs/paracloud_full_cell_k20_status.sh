#!/usr/bin/env bash
# Read-only status for the K=20 B/G seed-0 campaign.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
campaign="${1:-}"
[[ -n "$campaign" ]] || { echo "usage: $0 CAMPAIGN_DIR" >&2; exit 2; }
cd "$PHC2D_ROOT"
echo "checked_at=$(date -Is)"
echo "task_id,stage,status,last_epoch,last_val_mre"
for manifest in "$campaign/admission_manifest.csv" "$campaign/task_manifest.csv"; do
  [[ -f "$manifest" ]] || continue
  tail -n +2 "$manifest" | while IFS=, read -r task_index task_id model train_lane seed head grid_field config config_sha256 dataset_sha256 idx_train_hash idx_val_hash idx_test_hash k_points_sha256 run_dir rest; do
    status="missing"; epoch=""; mre=""
    [[ -f "$run_dir/results.json" ]] && status="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$run_dir/results.json" 2>/dev/null || echo parse_failed)"
    if [[ -f "$run_dir/history.csv" ]]; then
      read -r epoch mre < <(tail -n 1 "$run_dir/history.csv" | awk -F, '{print $1, $3}')
    fi
    echo "$task_id,$(basename "$manifest"),$status,$epoch,$mre"
  done
done
