#!/usr/bin/env bash
# Dry-run or submit one frozen Full-Wh upgrade 50k confirmation wave.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

mode="${1:-}"
input="${2:-}"
third="${3:-}"
fourth="${4:-}"
if [[ "$mode" != "dry-run" && "$mode" != "submit" ]]; then
  echo "usage: $0 dry-run HEAD_LOCK ADMISSION_LOCK AUTH | submit CAMPAIGN AUTH 50K-W1..50K-W4" >&2
  exit 2
fi
if [[ -z "$input" || -z "$third" || -z "$fourth" ]]; then
  echo "all mode-specific arguments are required" >&2
  exit 2
fi

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
cd "$PHC2D_ROOT"
actual_commit="$(tr -d '[:space:]' < DEPLOYED_GIT_COMMIT)"
expected_commit="${PHC2D_EXPECTED_COMMIT:-}"
if [[ -z "$expected_commit" || "$actual_commit" != "$expected_commit" ]]; then
  echo "deployment mismatch: deployed=$actual_commit expected=${expected_commit:-missing}" >&2
  exit 3
fi

binary_data=data/full_cell_binary_bz9_50000_xingyi_v1.h5
graded_data=data/full_cell_graded_bz9_50000_xingyi_v1.h5
binary_hash=e2d99e0c59c3bf5dfebc507e756dbcd18c8623f331ad6193832d774df5213f00
graded_hash=172786bc4b689d13f5938480fe82c25a4b8d9bad9a7d98bcd7eac778dad38a46
test "$(sha256sum "$binary_data" | cut -d' ' -f1)" = "$binary_hash"
test "$(sha256sum "$graded_data" | cut -d' ' -f1)" = "$graded_hash"
configs=(
  configs/research/full_cell_forward_upgrade_binary_50k_v1.yaml
  configs/research/full_cell_forward_upgrade_graded_50k_v1.yaml
  configs/research/full_cell_forward_upgrade_dual_lane_50k_v1.yaml
)
for config in "${configs[@]}"; do test -f "$config"; done

if [[ "$mode" == "dry-run" ]]; then
  head_lock="$input"
  admission_lock="$third"
  authorization_id="$fourth"
  timestamp="$(date +%Y%m%d%H%M%S)"
  campaign_dir="runs/$(date +%Y%m%d)/full_cell_upgrade_50k_dry-run_${timestamp}"
  python -m phc2d.forward.full_cell_upgrade_50k \
    --head-lock "$head_lock" \
    --admission-lock "$admission_lock" \
    --output-dir "$campaign_dir" \
    --repository-root "$PHC2D_ROOT"
  manifest="$campaign_dir/task_manifest.csv"
  {
    echo "mode=dry-run"
    echo "authorization_id=$authorization_id"
    echo "deployed_git_commit=$actual_commit"
    echo "campaign_dir=$campaign_dir"
    echo "manifest=$manifest"
    echo "manifest_sha256=$(sha256sum "$manifest" | cut -d' ' -f1)"
    echo "head_selection_lock=$head_lock"
    echo "head_selection_lock_sha256=$(sha256sum "$head_lock" | cut -d' ' -f1)"
    echo "admission_lock=$admission_lock"
    echo "admission_lock_sha256=$(sha256sum "$admission_lock" | cut -d' ' -f1)"
    echo "binary_dataset_sha256=$binary_hash"
    echo "graded_dataset_sha256=$graded_hash"
    for config in "${configs[@]}"; do key="$(basename "$config" .yaml)"; echo "config_${key}_sha256=$(sha256sum "$config" | cut -d' ' -f1)"; done
    echo "task_rows=27"
    echo "array_concurrency=8"
  } > "$campaign_dir/submission.env"
  echo "$campaign_dir"
  exit 0
fi

campaign_dir="$input"
submit_authorization_id="$third"
submit_wave="$fourth"
if [[ ! "$submit_wave" =~ ^50K-W[1-4]$ ]]; then
  echo "50k submit requires wave 50K-W1..50K-W4" >&2
  exit 2
fi
test -f "$campaign_dir/submission.env"
source "$campaign_dir/submission.env"
if [[ "$mode" != "dry-run" || "$deployed_git_commit" != "$expected_commit" ]]; then
  echo "50k dry-run campaign identity mismatch" >&2
  exit 3
fi
if [[ "${PHC2D_AUTHORIZE_UPGRADE_50K:-}" != "YES" || "$submit_authorization_id" != "${PHC2D_EXPECTED_50K_AUTHORIZATION_ID:-}" ]]; then
  echo "50k submit authorization mismatch" >&2
  exit 4
fi
test "$(sha256sum "$manifest" | cut -d' ' -f1)" = "$manifest_sha256"
test "$(sha256sum "$head_selection_lock" | cut -d' ' -f1)" = "$head_selection_lock_sha256"
test "$(sha256sum "$admission_lock" | cut -d' ' -f1)" = "$admission_lock_sha256"
for config in "${configs[@]}"; do
  key="$(basename "$config" .yaml)"
  expected_key="config_${key}_sha256"
  test "$(sha256sum "$config" | cut -d' ' -f1)" = "${!expected_key}"
done
wave_key="${submit_wave//-/_}"
if grep -q "^wave_${wave_key}_job_id=" "$campaign_dir/submission.env"; then
  echo "50k wave $submit_wave was already submitted" >&2
  exit 5
fi

case "$submit_wave" in
  50K-W1) previous_end=-1 ;;
  50K-W2) previous_end=7 ;;
  50K-W3) previous_end=15 ;;
  50K-W4) previous_end=23 ;;
esac
if (( previous_end >= 0 )); then
  for task_index in $(seq 0 "$previous_end"); do
    row="$(sed -n "$((task_index + 2))p" "$manifest")"; row="${row%$'\r'}"
    action="$(printf '%s' "$row" | cut -d, -f5)"
    [[ "$action" == "RUN" ]] || continue
    result_dir="$(printf '%s' "$row" | cut -d, -f11)"
    for artifact in run.log run.meta.json config.resolved.yaml results.json metrics.json history.csv model.pt checkpoint.pt model_summary.json epoch100.pt n40_profile.csv n40_profile.json; do
      test -s "$result_dir/$artifact" || { echo "previous 50k task $task_index missing $artifact" >&2; exit 5; }
    done
    grep -Eq 'formal_training_success|dual_lane_training_success' "$result_dir/results.json"
    grep -q 'profile_collected' "$result_dir/n40_profile.json"
  done
fi

indices="$(awk -F, -v wave="$submit_wave" 'NR>1 {gsub(/\r/, "", $0); if ($2==wave && $5=="RUN") print $1}' "$manifest" | paste -sd, -)"
if [[ -z "$indices" ]]; then
  echo "50k wave $submit_wave has no admitted RUN rows" >&2
  exit 5
fi
job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 24:00:00 \
  --array="${indices}%8" -J "fcup_${submit_wave}" \
  -o "$campaign_dir/slurm_%A_%a.out" -e "$campaign_dir/slurm_%A_%a.err" \
  --export="ALL,PHC2D_UPGRADE_50K_MANIFEST=$manifest" \
  jobs/paracloud_full_cell_upgrade_50k_train.sh)"
{
  echo "submitted_authorization_id=$submit_authorization_id"
  echo "wave_${wave_key}_job_id=$job_id"
  echo "wave_${wave_key}_indices=$indices"
  echo "wave_${wave_key}_submitted_at=$(date -Is)"
} >> "$campaign_dir/submission.env"
echo "$campaign_dir" > runs/full_cell_upgrade_active_50k.txt
echo "$campaign_dir"
