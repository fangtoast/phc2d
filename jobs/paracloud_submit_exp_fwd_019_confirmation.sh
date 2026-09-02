#!/usr/bin/env bash
# Gate and submit EXP-FWD-019 three-seed component confirmation on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"; cd "$PHC2D_ROOT"
export PYTHONPATH="$PHC2D_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
: "${PHC2D_EXPECTED_COMMIT:?Set PHC2D_EXPECTED_COMMIT.}"
: "${PHC2D_EXPECTED_FWD019_AUTHORIZATION_ID:?Set authorization identity.}"
[[ "${PHC2D_AUTHORIZE_EXP_FWD_019:-}" == "YES" ]] || { echo "authorization missing" >&2; exit 4; }
[[ "$(tr -d '[:space:]' < DEPLOYED_GIT_COMMIT)" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo "deployment mismatch" >&2; exit 3; }
source jobs/paracloud_exp_fwd_019_chunk_lib.sh
mode="${1:?usage: $0 SEED0_SCREEN_DIR | --existing-campaign CAMPAIGN START END}"
if [[ "$mode" == "--existing-campaign" ]]; then
  campaign="${2:?existing campaign is required}"
  start="${3:?chunk start is required}"
  end="${4:?chunk end is required}"
  screen_dir="$(sed -n 's/^screen_dir=//p' "$campaign/freeze.env")"
else
  screen_dir="$mode"
  campaign="runs/$(date +%Y%m%d)/EXP-FWD-019_confirmation_$(date +%Y%m%d%H%M%S)"
  "${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}/bin/python" -m phc2d.forward.component_campaign --screen-dir "$screen_dir" --output-dir "$campaign" --repository-root "$PHC2D_ROOT"
  printf 'screen_dir=%s\ndeployed_git_commit=%s\n' "$screen_dir" "$PHC2D_EXPECTED_COMMIT" > "$campaign/freeze.env"
  start=0
  end=15
fi
phc2d_fwd019_submit_chunk "$campaign" "$start" "$end" 42 24:00:00 fwd019_c3 three_seed_validation_only_no_test
printf 'screen_dir=%s\n' "$screen_dir" >> "$PHC2D_FWD019_SUBMISSION_EVIDENCE"
printf '%s\n' "$campaign" > runs/EXP-FWD-019_active_confirmation.txt
printf 'campaign=%s\njob_id=%s\narray=%s-%s%%3\n' "$campaign" "$PHC2D_FWD019_SUBMITTED_JOB_ID" "$start" "$end"
