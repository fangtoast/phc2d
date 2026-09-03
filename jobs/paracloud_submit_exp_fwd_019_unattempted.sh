#!/usr/bin/env bash
# Submit an unattempted tail slice from a partially completed EXP-FWD-019 screen.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"; cd "$PHC2D_ROOT"
export PYTHONPATH="$PHC2D_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
: "${PHC2D_EXPECTED_COMMIT:?Set PHC2D_EXPECTED_COMMIT.}"
: "${PHC2D_EXPECTED_FWD019_AUTHORIZATION_ID:?Set authorization identity.}"
: "${PHC2D_EXPECTED_SOURCE_JOB_ID:?Set the source array job id.}"
[[ "${PHC2D_AUTHORIZE_EXP_FWD_019:-}" == "YES" ]] || { echo "authorization missing" >&2; exit 4; }
[[ "$(tr -d '[:space:]' < DEPLOYED_GIT_COMMIT)" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo "deployment mismatch" >&2; exit 3; }
source_screen="${1:?usage: $0 SOURCE_SCREEN TASK_INDICES}"
task_indices="${2:?comma-separated source task indices are required}"
[[ "$task_indices" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "invalid task index list" >&2; exit 5; }
source_evidence="$source_screen/submission_chunk_000_015.env"
[[ "$(sed -n 's/^job_id=//p' "$source_evidence")" == "$PHC2D_EXPECTED_SOURCE_JOB_ID" ]] || { echo "source job mismatch" >&2; exit 6; }
source_end="$(sed -n 's/^array_end=//p' "$source_evidence")"
IFS=, read -r -a requested <<< "$task_indices"
(( ${#requested[@]} <= 16 )) || { echo "continuation slice exceeds N40 submit cap" >&2; exit 5; }
for source_index in "${requested[@]}"; do
  (( source_index > source_end )) || { echo "source task ${source_index} was included in the original array" >&2; exit 6; }
  row="$(sed -n "$((source_index + 2))p" "$source_screen/task_manifest.csv")"; row="${row%$'\r'}"
  IFS=, read -r _ _ _ _ _ _ _ _ source_run_dir <<< "$row"
  [[ -n "$source_run_dir" && ! -e "$source_run_dir" ]] || { echo "source task ${source_index} has an existing run path" >&2; exit 6; }
done
campaign="runs/$(date +%Y%m%d)/EXP-FWD-019_continuation_$(date +%Y%m%d%H%M%S)"
"${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}/bin/python" -m phc2d.forward.component_campaign \
  --recovery-screen-dir "$source_screen" --task-indices "$task_indices" \
  --output-dir "$campaign" --repository-root "$PHC2D_ROOT"
source jobs/paracloud_exp_fwd_019_chunk_lib.sh
last=$((${#requested[@]} - 1))
phc2d_fwd019_submit_chunk "$campaign" 0 "$last" "${#requested[@]}" 08:00:00 fwd019_tail seed0_validation_only_no_test_unattempted
printf 'source_screen=%s\nsource_job_id=%s\nsource_task_indices=%s\nsource_array_end=%s\n' \
  "$source_screen" "$PHC2D_EXPECTED_SOURCE_JOB_ID" "$task_indices" "$source_end" >> "$PHC2D_FWD019_SUBMISSION_EVIDENCE"
printf '%s\n' "$campaign" > runs/EXP-FWD-019_active_continuation.txt
printf 'campaign=%s\njob_id=%s\narray=0-%s%%3\n' "$campaign" "$PHC2D_FWD019_SUBMITTED_JOB_ID" "$last"
