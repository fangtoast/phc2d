#!/usr/bin/env bash
# Freeze and submit EXP-FWD-019 C0 validation-only diagnostics.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
cd "$PHC2D_ROOT"
: "${PHC2D_EXPECTED_COMMIT:?Set PHC2D_EXPECTED_COMMIT.}"
: "${PHC2D_EXPECTED_FWD019_AUTHORIZATION_ID:?Set authorization identity.}"
[[ "${PHC2D_AUTHORIZE_EXP_FWD_019:-}" == "YES" ]] || { echo "EXP-FWD-019 authorization missing" >&2; exit 4; }
[[ "$(tr -d '[:space:]' < DEPLOYED_GIT_COMMIT)" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo "deployment mismatch" >&2; exit 3; }
test "$(sha256sum data/full_cell_binary_bz9_50000_xingyi_v1.h5 | cut -d' ' -f1)" = e2d99e0c59c3bf5dfebc507e756dbcd18c8623f331ad6193832d774df5213f00
test "$(sha256sum data/full_cell_graded_bz9_50000_xingyi_v1.h5 | cut -d' ' -f1)" = 172786bc4b689d13f5938480fe82c25a4b8d9bad9a7d98bcd7eac778dad38a46
campaign="runs/$(date +%Y%m%d)/EXP-FWD-019_C0_$(date +%Y%m%d%H%M%S)"
mkdir -p "$campaign"
manifest="$campaign/c0_manifest.csv"
printf '%s\n' 'task_index,case_id,model,lane,config,checkpoint,output_dir' > "$manifest"
rows=(
  'fno_b,fno2d_band_v2,binary,runs/20260807/fc50k_binary_native_n40000_e200_train_20260807163009_fno2d_band_v2_seed0'
  'fno_g,fno2d_band_v2,graded,runs/20260807/fc50k_graded_continuous_n40000_e200_train_20260807195617_fno2d_band_v2_seed0'
  'tfno_b,tfno2d_band_v3,binary,runs/20260809/full_cell_upgrade_50k_dry-run_20260809134906/50K-W1_TF_D_B_s0'
  'tfno_g,tfno2d_band_v3,graded,runs/20260809/full_cell_upgrade_50k_dry-run_20260809134906/50K-W1_TF_D_G_s0'
  'convnext_b,convnext_band_v1,binary,runs/20260809/full_cell_upgrade_50k_dry-run_20260809134906/50K-W3_CX_D_B_s0'
  'convnext_g,convnext_band_v1,graded,runs/20260809/full_cell_upgrade_50k_dry-run_20260809134906/50K-W4_CX_D_G_s0'
  'joint,dual_lane_band_operator_v1,joint,runs/20260809/full_cell_upgrade_50k_dry-run_20260809134906/50K-W1_DL_D_J_s0'
)
index=0
for item in "${rows[@]}"; do
  IFS=, read -r case_id model lane source_dir <<< "$item"
  test -s "$source_dir/model.pt"; test -s "$source_dir/config.resolved.yaml"
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$index" "$case_id" "$model" "$lane" "$source_dir/config.resolved.yaml" "$source_dir/model.pt" "$campaign/$case_id" >> "$manifest"
  index=$((index + 1))
done
job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 02:00:00 --array=0-6%3 \
  -J fwd019_c0 -o "$campaign/slurm_%A_%a.out" -e "$campaign/slurm_%A_%a.err" \
  --export="ALL,PHC2D_FWD019_C0_MANIFEST=$manifest" jobs/paracloud_exp_fwd_019_c0.sh)"
printf 'authorization_id=%s\ndeployed_git_commit=%s\nmanifest=%s\nmanifest_sha256=%s\njob_id=%s\nsubmission_scope=validation_only_no_test\n' \
  "$PHC2D_EXPECTED_FWD019_AUTHORIZATION_ID" "$PHC2D_EXPECTED_COMMIT" "$manifest" "$(sha256sum "$manifest" | cut -d' ' -f1)" "$job_id" > "$campaign/submission.env"
printf '%s\n' "$campaign" > runs/EXP-FWD-019_active.txt
echo "$campaign"
