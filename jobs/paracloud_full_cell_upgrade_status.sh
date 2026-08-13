#!/usr/bin/env bash
# Read-only status for one active upgraded Full-Wh N40 wave.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

campaign_dir="${1:-}"
if [[ -z "$campaign_dir" ]]; then
  pointer="runs/full_cell_upgrade_active_wave.txt"
  [[ -f "$pointer" ]] || { echo "status=no_active_upgrade_wave"; exit 0; }
  campaign_dir="$(tr -d '\r\n' < "$pointer")"
fi
source "$campaign_dir/submission.env"
echo "checked_at=$(date -Is)"
echo "wave=$wave"
echo "campaign_dir=$campaign_dir"
echo "array_job_id=${array_job_id:-not_submitted}"
echo "task_id,job_id,state,elapsed,node,artifact_status,last_epoch,last_val_mre"
tail -n +2 "$manifest" | while IFS=, read -r task_index task_id model head lane seed config run_dir admission; do
  job_ref="${array_job_id:-dry_run}_${task_index}"
  state="DRY_RUN"
  elapsed=""
  node=""
  if [[ -n "${array_job_id:-}" ]]; then
    row="$(sacct -n -X -j "$job_ref" -o State,Elapsed,NodeList -P 2>/dev/null | head -n 1 || true)"
    state="$(printf '%s' "$row" | cut -d'|' -f1)"
    elapsed="$(printf '%s' "$row" | cut -d'|' -f2)"
    node="$(printf '%s' "$row" | cut -d'|' -f3)"
  fi
  artifact_status="missing"
  last_epoch=""
  last_val_mre=""
  if [[ -f "$run_dir/results.json" ]]; then
    artifact_status="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$run_dir/results.json" 2>/dev/null || echo parse_failed)"
  elif [[ -f "$run_dir/history.csv" ]]; then
    artifact_status="running_artifacts"
  fi
  if [[ -f "$run_dir/history.csv" ]]; then
    values="$(python -c 'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); print((r[-1].get("epoch","")+"|"+r[-1].get("val_bands_2_10_mre_pct","")) if r else "|")' "$run_dir/history.csv" 2>/dev/null || echo '|')"
    last_epoch="${values%%|*}"
    last_val_mre="${values#*|}"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$task_id" "$job_ref" "${state:-UNKNOWN}" "$elapsed" "$node" "$artifact_status" "$last_epoch" "$last_val_mre"
done
