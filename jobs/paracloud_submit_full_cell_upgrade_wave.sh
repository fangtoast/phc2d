#!/usr/bin/env bash
# Build or submit one frozen upgraded Full-Wh admission/10k wave on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

mode="${1:-}"
wave="${2:-}"
authorization_id="${3:-}"
if [[ "$mode" != "dry-run" && "$mode" != "submit" ]]; then
  echo "usage: $0 dry-run|submit A0|A1|A2|10K-W1..10K-W7 AUTHORIZATION_ID" >&2
  exit 2
fi
if [[ -z "$wave" || -z "$authorization_id" ]]; then
  echo "wave and authorization id are required" >&2
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
if [[ "$mode" == "submit" && "${PHC2D_AUTHORIZE_UPGRADE_CAMPAIGN:-}" != "YES" ]]; then
  echo "submit requires PHC2D_AUTHORIZE_UPGRADE_CAMPAIGN=YES" >&2
  exit 4
fi

binary_config="configs/research/full_cell_forward_upgrade_binary_10k_v1.yaml"
graded_config="configs/research/full_cell_forward_upgrade_graded_10k_v1.yaml"
joint_config="configs/research/full_cell_forward_upgrade_dual_lane_10k_v1.yaml"
for config in "$binary_config" "$graded_config" "$joint_config"; do
  test -f "$config" || { echo "missing config: $config" >&2; exit 3; }
done
binary_data="data/full_cell_binary_bz9_10000_xingyi_v1.h5"
graded_data="data/full_cell_graded_bz9_10000_paracloud_m9_v1.h5"
test "$(sha256sum "$binary_data" | awk '{print $1}')" = "1f131f2bda7ccb1f1736809bdf6bafd534096cb275e22ce3e56fc5d910530c2f"
test "$(sha256sum "$graded_data" | awk '{print $1}')" = "d16d838e098215c05dc299bf7c4bbc8c6bc00f6ef35b9772546aa4e7c2b8b728"

case "$wave" in
  A0) specs=(TF:D:B:0 TF:D:G:0 TF:Q:B:0 TF:Q:G:0 DL:D:J:0 DL:Q:J:0 MM:D:B:0 MM:D:G:0) ;;
  A1) specs=(MM:Q:B:0 MM:Q:G:0 RM:D:B:0 RM:D:G:0 RM:Q:B:0 RM:Q:G:0 CX:D:B:0 CX:D:G:0) ;;
  A2) specs=(CX:Q:B:0 CX:Q:G:0) ;;
  10K-W1) specs=(TF:D:B:0 TF:D:B:1 TF:D:B:2 TF:D:G:0 TF:D:G:1 TF:D:G:2 DL:D:J:0 DL:D:J:1) ;;
  10K-W2) specs=(DL:D:J:2 TF:Q:B:0 TF:Q:B:1 TF:Q:B:2 TF:Q:G:0 TF:Q:G:1 TF:Q:G:2 DL:Q:J:0) ;;
  10K-W3) specs=(DL:Q:J:1 DL:Q:J:2 MM:D:B:0 MM:D:B:1 MM:D:B:2 MM:D:G:0 MM:D:G:1 MM:D:G:2) ;;
  10K-W4) specs=(MM:Q:B:0 MM:Q:B:1 MM:Q:B:2 MM:Q:G:0 MM:Q:G:1 MM:Q:G:2 RM:D:B:0 RM:D:B:1) ;;
  10K-W5) specs=(RM:D:B:2 RM:D:G:0 RM:D:G:1 RM:D:G:2 RM:Q:B:0 RM:Q:B:1 RM:Q:B:2 RM:Q:G:0) ;;
  10K-W6) specs=(RM:Q:G:1 RM:Q:G:2 CX:D:B:0 CX:D:B:1 CX:D:B:2 CX:D:G:0 CX:D:G:1 CX:D:G:2) ;;
  10K-W7) specs=(CX:Q:B:0 CX:Q:B:1 CX:Q:B:2 CX:Q:G:0 CX:Q:G:1 CX:Q:G:2) ;;
  *) echo "unknown or not-yet-admitted wave: $wave" >&2; exit 2 ;;
esac

model_name() {
  case "$1" in
    MM) echo mlp_mixer_band_v1 ;;
    RM) echo res_mlp_gated_v2 ;;
    CX) echo convnext_band_v1 ;;
    TF) echo tfno2d_band_v3 ;;
    DL) echo dual_lane_band_operator_v1 ;;
  esac
}
head_name() { [[ "$1" == "D" ]] && echo dense || echo k_conditioned; }
config_name() {
  case "$1" in B) echo "$binary_config" ;; G) echo "$graded_config" ;; J) echo "$joint_config" ;; esac
}

timestamp="$(date +%Y%m%d%H%M%S)"
campaign_dir="runs/$(date +%Y%m%d)/full_cell_upgrade_${wave}_${mode}_${timestamp}"
mkdir -p "$campaign_dir"
manifest="$campaign_dir/task_manifest.csv"
printf 'task_index,task_id,model,head,lane,seed,config,run_dir,admission\n' > "$manifest"
admission=0
[[ "$wave" == A* ]] && admission=1
index=0
for spec in "${specs[@]}"; do
  IFS=: read -r short_model short_head lane seed <<< "$spec"
  model="$(model_name "$short_model")"
  head="$(head_name "$short_head")"
  config="$(config_name "$lane")"
  task_id="${wave}_${short_model}_${short_head}_${lane}_s${seed}"
  run_dir="$campaign_dir/${task_id}"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$index" "$task_id" "$model" "$head" "$lane" "$seed" "$config" "$run_dir" "$admission" >> "$manifest"
  index=$((index + 1))
done
manifest_hash="$(sha256sum "$manifest" | awk '{print $1}')"
{
  echo "wave=$wave"
  echo "mode=$mode"
  echo "authorization_id=$authorization_id"
  echo "deployed_git_commit=$deployed_commit"
  echo "manifest=$manifest"
  echo "manifest_sha256=$manifest_hash"
  echo "task_count=${#specs[@]}"
  echo "array_concurrency=8"
} > "$campaign_dir/submission.env"

if [[ "$mode" == "dry-run" ]]; then
  echo "$campaign_dir"
  exit 0
fi

time_limit="12:00:00"
[[ "$admission" == "1" ]] && time_limit="00:20:00"
array_max=$((${#specs[@]} - 1))
job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t "$time_limit" \
  --array="0-${array_max}%8" -J "fcup_${wave}" \
  -o "$campaign_dir/slurm_%A_%a.out" -e "$campaign_dir/slurm_%A_%a.err" \
  --export="ALL,PHC2D_UPGRADE_MANIFEST=$manifest" \
  jobs/paracloud_full_cell_upgrade_train.sh)"
echo "array_job_id=$job_id" >> "$campaign_dir/submission.env"
echo "$campaign_dir" > runs/full_cell_upgrade_active_wave.txt
echo "$campaign_dir"
