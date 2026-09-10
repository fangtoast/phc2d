#!/usr/bin/env bash
# Read-only EXP-FWD-007 N40 campaign status snapshot; safe for 30-minute monitoring.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
if [[ "${1:-}" == "--campaign-dir" ]]; then
  campaign="${2:?usage: $0 [--campaign-dir] CAMPAIGN_DIR}"
else
  campaign="${1:?usage: $0 [--campaign-dir] CAMPAIGN_DIR}"
fi
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"; PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
echo "checked_at=$(date -Is)"
manifests=()
for manifest in "$campaign/admission_a_manifest.csv" "$campaign/admission_b_manifest.csv"; do
  [[ -f "$manifest" ]] && manifests+=("$manifest")
done
[[ ${#manifests[@]} -gt 0 ]] || [[ ! -f "$campaign/admission_manifest.csv" ]] || manifests+=("$campaign/admission_manifest.csv")
manifests+=("$campaign/task_manifest.csv")
for manifest in "${manifests[@]}"; do
  [[ -f "$manifest" ]] || continue
  tail -n +2 "$manifest" | while IFS=, read -r task_index task_id model train_lane wave seed head config rest; do
    run_dir="$(awk -F, -v id="$task_id" 'NR>1 && $2==id {print $15}' "$manifest")"; status="missing"
    [[ -f "$run_dir/results.json" ]] && status="$($PHC2D_ENV/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$run_dir/results.json" 2>/dev/null || echo parse_failed)"
    echo "$task_id,$wave,$status"
  done
done
command -v squeue >/dev/null && squeue -u "$USER" -o '%.18i %.24j %.10T %.12M %.12l %R' || true
if command -v sacct >/dev/null && [[ -f "$campaign/submission.env" ]]; then
  job_ids="$(sed -n 's/^job_id=//p' "$campaign/submission.env" | paste -sd, -)"
  [[ -z "$job_ids" ]] || sacct -j "$job_ids" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS -P || true
fi
