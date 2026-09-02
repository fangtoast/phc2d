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
screen_dir="${1:?usage: $0 SEED0_SCREEN_DIR}"
campaign="runs/$(date +%Y%m%d)/EXP-FWD-019_confirmation_$(date +%Y%m%d%H%M%S)"
"${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}/bin/python" -m phc2d.forward.component_campaign --screen-dir "$screen_dir" --output-dir "$campaign" --repository-root "$PHC2D_ROOT"
manifest="$campaign/task_manifest.csv"; count="$(($(wc -l < "$manifest") - 1))"
if (( count == 0 )); then echo "$campaign"; exit 0; fi
last=$((count - 1))
job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 24:00:00 --array="0-${last}%3" \
  -J fwd019_c3 -o "$campaign/slurm_%A_%a.out" -e "$campaign/slurm_%A_%a.err" \
  --export="ALL,PHC2D_FWD019_SCREEN_MANIFEST=$manifest" jobs/paracloud_exp_fwd_019_seed0_train.sh)"
printf 'authorization_id=%s\ndeployed_git_commit=%s\nscreen_dir=%s\nmanifest=%s\nmanifest_sha256=%s\njob_id=%s\nsubmission_scope=three_seed_validation_only_no_test\n' \
  "$PHC2D_EXPECTED_FWD019_AUTHORIZATION_ID" "$PHC2D_EXPECTED_COMMIT" "$screen_dir" "$manifest" "$(sha256sum "$manifest" | cut -d' ' -f1)" "$job_id" > "$campaign/submission.env"
printf '%s\n' "$campaign" > runs/EXP-FWD-019_active_confirmation.txt
echo "$campaign"
