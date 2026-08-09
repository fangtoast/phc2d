#!/usr/bin/env bash
# Dry-run or submit the 51-checkpoint frozen-head Full-Wh native-test campaign.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

mode="${1:-}"
input="${2:-}"
authorization_id="${3:-}"
test_wave="${4:-}"
if [[ "$mode" != "dry-run" && "$mode" != "submit" ]]; then
  echo "usage: $0 dry-run HEAD_LOCK AUTHORIZATION_ID | submit DRY_RUN_CAMPAIGN AUTHORIZATION_ID T1..T7" >&2
  exit 2
fi
if [[ "$mode" == "submit" && ! "$test_wave" =~ ^T[1-7]$ ]]; then
  echo "test submit requires wave T1..T7" >&2
  exit 2
fi
if [[ -z "$input" || -z "$authorization_id" ]]; then
  echo "head lock or dry-run campaign and authorization id are required" >&2
  exit 2
fi

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
cd "$PHC2D_ROOT"
deployed_commit="$(tr -d '\r\n' < DEPLOYED_GIT_COMMIT)"
expected_commit="${PHC2D_EXPECTED_COMMIT:-}"
if [[ -z "$expected_commit" || "$deployed_commit" != "$expected_commit" ]]; then
  echo "deployment mismatch: deployed=$deployed_commit expected=${expected_commit:-missing}" >&2
  exit 3
fi

if [[ "$mode" == "dry-run" ]]; then
  head_lock="$input"
  test -f "$head_lock" || { echo "missing head-selection lock: $head_lock" >&2; exit 3; }
  timestamp="$(date +%Y%m%d%H%M%S)"
  campaign_dir="runs/$(date +%Y%m%d)/full_cell_upgrade_10k_frozen_test_dry-run_${timestamp}"
  python -m phc2d.forward.full_cell_upgrade_test plan \
    --head-lock "$head_lock" \
    --baseline-binary-manifest runs/20260806/full_cell_forward_binary_native_v1_wave_aggregation_20260806/run_manifest.csv \
    --baseline-graded-manifest runs/20260806/full_cell_forward_graded_continuous_v1_wave_aggregation_20260806/run_manifest.csv \
    --output-dir "$campaign_dir" \
    --repository-root "$PHC2D_ROOT"
  manifest="$campaign_dir/test_manifest.csv"
  {
    echo "mode=dry-run"
    echo "authorization_id=$authorization_id"
    echo "deployed_git_commit=$deployed_commit"
    echo "campaign_dir=$campaign_dir"
    echo "manifest=$manifest"
    echo "manifest_sha256=$(sha256sum "$manifest" | awk '{print $1}')"
    echo "head_selection_lock=$head_lock"
    echo "head_selection_lock_sha256=$(sha256sum "$head_lock" | awk '{print $1}')"
    echo "task_count=51"
    echo "array_concurrency=8"
  } > "$campaign_dir/submission.env"
  echo "$campaign_dir"
  exit 0
fi

campaign_dir="$input"
test -f "$campaign_dir/submission.env" || { echo "missing frozen dry-run submission.env" >&2; exit 3; }
submit_authorization_id="$authorization_id"
source "$campaign_dir/submission.env"
if [[ "$mode" != "dry-run" || "$deployed_git_commit" != "$expected_commit" ]]; then
  echo "dry-run campaign commit or mode mismatch" >&2
  exit 3
fi
if [[ "$submit_authorization_id" != "${PHC2D_EXPECTED_TEST_AUTHORIZATION_ID:-}" || -z "$submit_authorization_id" ]]; then
  echo "test submit authorization id mismatch" >&2
  exit 4
fi
if [[ "${PHC2D_AUTHORIZE_UPGRADE_TEST:-}" != "YES" ]]; then
  echo "submit requires PHC2D_AUTHORIZE_UPGRADE_TEST=YES" >&2
  exit 4
fi
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$manifest_sha256"
test "$(sha256sum "$head_selection_lock" | awk '{print $1}')" = "$head_selection_lock_sha256"
test "$(($(wc -l < "$manifest") - 1))" = "51"

case "$test_wave" in
  T1) array_start=0; array_end=7; previous_end=-1 ;;
  T2) array_start=8; array_end=15; previous_end=7 ;;
  T3) array_start=16; array_end=23; previous_end=15 ;;
  T4) array_start=24; array_end=31; previous_end=23 ;;
  T5) array_start=32; array_end=39; previous_end=31 ;;
  T6) array_start=40; array_end=47; previous_end=39 ;;
  T7) array_start=48; array_end=50; previous_end=47 ;;
esac
if grep -q "^test_wave_${test_wave}_job_id=" "$campaign_dir/submission.env"; then
  echo "test wave $test_wave was already submitted" >&2
  exit 5
fi
if (( previous_end >= 0 )); then
  for task_index in $(seq 0 "$previous_end"); do
    row="$(sed -n "$((task_index + 2))p" "$manifest")"
    result_dir="$(printf '%s' "$row" | cut -d, -f11)"
    test -f "$result_dir/results.json" || { echo "previous test task $task_index missing results.json" >&2; exit 5; }
    test -f "$result_dir/n40_profile.json" || { echo "previous test task $task_index missing N40 profile" >&2; exit 5; }
    python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "frozen_checkpoint_test_success"' "$result_dir/results.json"
  done
fi

job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 02:00:00 \
  --array="${array_start}-${array_end}%8" -J "fcup_${test_wave}" \
  -o "$campaign_dir/slurm_%A_%a.out" -e "$campaign_dir/slurm_%A_%a.err" \
  --export="ALL,PHC2D_UPGRADE_TEST_MANIFEST=$manifest" \
  jobs/paracloud_full_cell_upgrade_test.sh)"
{
  echo "submitted_authorization_id=$submit_authorization_id"
  echo "test_wave_${test_wave}_job_id=$job_id"
  echo "test_wave_${test_wave}_range=${array_start}-${array_end}"
  echo "test_wave_${test_wave}_submitted_at=$(date -Is)"
} >> "$campaign_dir/submission.env"
echo "$campaign_dir" > runs/full_cell_upgrade_active_test.txt
echo "$campaign_dir"
