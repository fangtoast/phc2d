#!/usr/bin/env bash
# Freeze and submit the gate-selected EXP-FWD-019 seed0 screen.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"; cd "$PHC2D_ROOT"
: "${PHC2D_EXPECTED_COMMIT:?Set PHC2D_EXPECTED_COMMIT.}"
: "${PHC2D_EXPECTED_FWD019_AUTHORIZATION_ID:?Set authorization identity.}"
[[ "${PHC2D_AUTHORIZE_EXP_FWD_019:-}" == "YES" ]] || { echo "authorization missing" >&2; exit 4; }
[[ "$(tr -d '[:space:]' < DEPLOYED_GIT_COMMIT)" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo "deployment mismatch" >&2; exit 3; }
c0_dir="${1:?usage: $0 C0_DIR}"
screen="runs/$(date +%Y%m%d)/EXP-FWD-019_seed0_$(date +%Y%m%d%H%M%S)"
python -m phc2d.forward.component_campaign --c0-dir "$c0_dir" --output-dir "$screen" --repository-root "$PHC2D_ROOT"
manifest="$screen/task_manifest.csv"; count="$(($(wc -l < "$manifest") - 1))"
[[ "$count" == 21 ]] || { echo "unexpected seed0 task count: $count" >&2; exit 5; }
job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 08:00:00 --array=0-20%3 \
  -J fwd019_s0 -o "$screen/slurm_%A_%a.out" -e "$screen/slurm_%A_%a.err" \
  --export="ALL,PHC2D_FWD019_SCREEN_MANIFEST=$manifest" jobs/paracloud_exp_fwd_019_seed0_train.sh)"
printf 'authorization_id=%s\ndeployed_git_commit=%s\nc0_dir=%s\nmanifest=%s\nmanifest_sha256=%s\njob_id=%s\nsubmission_scope=seed0_validation_only_no_test\n' \
  "$PHC2D_EXPECTED_FWD019_AUTHORIZATION_ID" "$PHC2D_EXPECTED_COMMIT" "$c0_dir" "$manifest" "$(sha256sum "$manifest" | cut -d' ' -f1)" "$job_id" > "$screen/submission.env"
printf '%s\n' "$screen" > runs/EXP-FWD-019_active_seed0.txt
echo "$screen"
