#!/usr/bin/env bash
# Read-only status summary for the currently active full-cell forward wave.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

manifest="${1:-}"
if [[ -z "$manifest" ]]; then
  pointer="runs/full_cell_forward_active_wave.txt"
  if [[ ! -f "$pointer" ]]; then
    echo "status=no_active_wave_pointer"
    exit 0
  fi
  manifest="$(tr -d '\r\n' < "$pointer")"
fi
if [[ ! -f "$manifest" ]]; then
  echo "status=manifest_missing"
  echo "manifest=$manifest"
  exit 0
fi

echo "checked_at=$(date -Is)"
echo "manifest=$manifest"
echo "model,seed,job_id,slurm_state,elapsed,node,artifact_status,last_epoch,last_val_mre"
tail -n +2 "$manifest" | while IFS=, read -r model seed job_id run_dir mode; do
  state_row="$(sacct -n -X -j "$job_id" -o State,Elapsed,NodeList -P 2>/dev/null | head -n 1 || true)"
  state="$(printf '%s' "$state_row" | cut -d'|' -f1)"
  elapsed="$(printf '%s' "$state_row" | cut -d'|' -f2)"
  node="$(printf '%s' "$state_row" | cut -d'|' -f3)"
  artifact_status="missing"
  last_epoch=""
  last_val_mre=""
  if [[ -f "$run_dir/results.json" ]]; then
    artifact_status="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$run_dir/results.json" 2>/dev/null || echo parse_failed)"
  elif [[ -f "$run_dir/history.csv" ]]; then
    artifact_status="running_artifacts"
  fi
  if [[ -f "$run_dir/history.csv" ]]; then
    history_values="$(python -c 'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); print((r[-1].get("epoch","")+"|"+r[-1].get("val_bands_2_10_mre_pct","")) if r else "|")' "$run_dir/history.csv" 2>/dev/null || echo '|')"
    last_epoch="${history_values%%|*}"
    last_val_mre="${history_values#*|}"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$model" "$seed" "$job_id" "${state:-UNKNOWN}" "${elapsed:-}" "${node:-}" \
    "$artifact_status" "$last_epoch" "$last_val_mre"
done
