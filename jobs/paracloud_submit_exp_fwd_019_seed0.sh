#!/usr/bin/env bash
# Freeze and submit one bounded chunk of the gate-selected EXP-FWD-019 seed0 screen.
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
mode="${1:?usage: $0 C0_DIR | --existing-screen SCREEN START END}"
if [[ "$mode" == "--existing-screen" ]]; then
  screen="${2:?existing screen is required}"
  start="${3:?chunk start is required}"
  end="${4:?chunk end is required}"
  c0_dir="$(sed -n 's/^c0_dir=//p' "$screen/freeze.env" 2>/dev/null || true)"
else
  c0_dir="$mode"
  screen="runs/$(date +%Y%m%d)/EXP-FWD-019_seed0_$(date +%Y%m%d%H%M%S)"
  "${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}/bin/python" -m phc2d.forward.component_campaign --c0-dir "$c0_dir" --output-dir "$screen" --repository-root "$PHC2D_ROOT"
  printf 'c0_dir=%s\ndeployed_git_commit=%s\n' "$c0_dir" "$PHC2D_EXPECTED_COMMIT" > "$screen/freeze.env"
  start=0
  end=15
fi
phc2d_fwd019_submit_chunk "$screen" "$start" "$end" 21 08:00:00 fwd019_s0 seed0_validation_only_no_test
printf 'c0_dir=%s\n' "$c0_dir" >> "$PHC2D_FWD019_SUBMISSION_EVIDENCE"
printf '%s\n' "$screen" > runs/EXP-FWD-019_active_seed0.txt
printf 'screen=%s\njob_id=%s\narray=%s-%s%%3\n' "$screen" "$PHC2D_FWD019_SUBMITTED_JOB_ID" "$start" "$end"
