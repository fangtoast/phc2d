#!/usr/bin/env bash
# Read-only status summary for a full-cell inverse Stage-A wave.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

manifest="${1:-}"
if [[ -z "$manifest" ]]; then
  pointer="runs/full_cell_inverse_stage_a_active_wave.txt"
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
echo "variant,method,seed,job_id,dependency,slurm_state,elapsed,node,artifact_status,last_epoch,last_val_total_loss,case_count,forward_surrogate_status"
tail -n +2 "$manifest" | while IFS=, read -r variant method seed job_id dependency run_dir mode submit_state; do
  state="DRY_RUN"
  elapsed=""
  node=""
  if [[ "$submit_state" != "dry_run" ]]; then
    state_row="$(sacct -n -X -j "$job_id" -o State,Elapsed,NodeList -P 2>/dev/null | head -n 1 || true)"
    state="$(printf '%s' "$state_row" | cut -d'|' -f1)"
    elapsed="$(printf '%s' "$state_row" | cut -d'|' -f2)"
    node="$(printf '%s' "$state_row" | cut -d'|' -f3)"
  fi
  artifact_status="missing"
  last_epoch=""
  last_val_total_loss=""
  case_count=""
  forward_status=""
  if [[ -f "$run_dir/results.json" ]]; then
    result_values="$(python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(str(p.get("status",""))+"|"+str(p.get("forward_surrogate_status","")))' "$run_dir/results.json" 2>/dev/null || echo 'parse_failed|')"
    artifact_status="${result_values%%|*}"
    forward_status="${result_values#*|}"
  elif [[ -f "$run_dir/history.csv" ]]; then
    artifact_status="running_artifacts"
  fi
  if [[ -f "$run_dir/history.csv" ]]; then
    history_values="$(python -c 'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); print((r[-1].get("epoch","")+"|"+r[-1].get("val_total_loss","")) if r else "|")' "$run_dir/history.csv" 2>/dev/null || echo '|')"
    last_epoch="${history_values%%|*}"
    last_val_total_loss="${history_values#*|}"
  fi
  if [[ -f "$run_dir/metrics.json" ]]; then
    case_count="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("case_count",""))' "$run_dir/metrics.json" 2>/dev/null || true)"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$variant" "$method" "$seed" "$job_id" "$dependency" "${state:-UNKNOWN}" "$elapsed" "$node" \
    "$artifact_status" "$last_epoch" "$last_val_total_loss" "$case_count" "$forward_status"
done
