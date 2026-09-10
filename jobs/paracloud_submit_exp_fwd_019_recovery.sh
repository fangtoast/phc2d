#!/usr/bin/env bash
# Freeze and submit one evidence-preserving EXP-FWD-019 correction slice on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"; cd "$PHC2D_ROOT"
export PYTHONPATH="$PHC2D_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
: "${PHC2D_EXPECTED_COMMIT:?Set PHC2D_EXPECTED_COMMIT.}"
: "${PHC2D_EXPECTED_FWD019_AUTHORIZATION_ID:?Set authorization identity.}"
: "${PHC2D_EXPECTED_SOURCE_JOB_ID:?Set the failed source array job id.}"
[[ "${PHC2D_AUTHORIZE_EXP_FWD_019:-}" == "YES" ]] || { echo "authorization missing" >&2; exit 4; }
[[ "$(tr -d '[:space:]' < DEPLOYED_GIT_COMMIT)" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo "deployment mismatch" >&2; exit 3; }
source_screen="${1:?usage: $0 SOURCE_SCREEN TASK_INDICES}"
task_indices="${2:?comma-separated source task indices are required}"
[[ "$task_indices" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "invalid task index list" >&2; exit 5; }
IFS=, read -r -a requested <<< "$task_indices"
(( ${#requested[@]} <= 16 )) || { echo "recovery slice exceeds N40 submit cap" >&2; exit 5; }
for source_index in "${requested[@]}"; do
  state="$(sacct -j "${PHC2D_EXPECTED_SOURCE_JOB_ID}_${source_index}" -X -n -P -o State | sed '/^[[:space:]]*$/d')"
  [[ "$state" == "FAILED" ]] || { echo "source task ${source_index} is not FAILED: $state" >&2; exit 6; }
done
campaign="runs/$(date +%Y%m%d)/EXP-FWD-019_recovery_$(date +%Y%m%d%H%M%S)"
"${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}/bin/python" -m phc2d.forward.component_campaign \
  --recovery-screen-dir "$source_screen" --task-indices "$task_indices" \
  --output-dir "$campaign" --repository-root "$PHC2D_ROOT"
source jobs/paracloud_exp_fwd_019_chunk_lib.sh
last=$((${#requested[@]} - 1))
phc2d_fwd019_submit_chunk "$campaign" 0 "$last" "${#requested[@]}" 08:00:00 fwd019_fix seed0_validation_only_no_test_correction
printf 'source_screen=%s\nsource_job_id=%s\nsource_task_indices=%s\n' \
  "$source_screen" "$PHC2D_EXPECTED_SOURCE_JOB_ID" "$task_indices" >> "$PHC2D_FWD019_SUBMISSION_EVIDENCE"
printf '%s\n' "$campaign" > runs/EXP-FWD-019_active_recovery.txt
printf 'campaign=%s\njob_id=%s\narray=0-%s%%3\n' "$campaign" "$PHC2D_FWD019_SUBMITTED_JOB_ID" "$last"
