#!/usr/bin/env bash
# Submit or inspect the frozen EXP-FWD-020 exploratory G3 matrix on N56.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
stage="${1:?usage: $0 preflight|submit|status|finalize CAMPAIGN_DIR}"
campaign="${2:?set frozen G3 campaign directory}"
PHC2D_ROOT="${PHC2D_ROOT:?set read-only deployment directory}"
PHC2D_ENV="${PHC2D_ENV:?set versioned N56 environment}"
: "${PHC2D_EXPECTED_COMMIT:?set deployed commit}"; : "${PHC2D_FWD020_AUTHORIZATION_ID:?set authorization identity}"
export PYTHONPATH="$PHC2D_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
[[ "$(tr -d '[:space:]' < "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT")" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo "deployment mismatch" >&2; exit 3; }
[[ "${PHC2D_AUTHORIZE_EXP_FWD_020:-}" == YES ]] || { echo "authorization missing" >&2; exit 4; }

case "$stage" in
  preflight)
    [[ ! -e "$campaign/preflight_submission.env" ]] || { echo "G3 preflight already submitted; no retry is permitted" >&2; exit 5; }
    job_id="$(sbatch --parsable -p gpu_5090 --gres=gpu:1 -c 8 -t 02:00:00 -J fwd020_g3_preflight \
      -o "$campaign/slurm_preflight_%j.out" -e "$campaign/slurm_preflight_%j.err" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_ENV=$PHC2D_ENV,PHC2D_G3_CAMPAIGN=$campaign,PHC2D_G3_STAGE=preflight" \
      "$PHC2D_ROOT/jobs/paracloud_exp_fwd_020_g3.sh)"
    printf 'authorization_id=%s\ndeployed_git_commit=%s\njob_id=%s\nstage=preflight\n' "$PHC2D_FWD020_AUTHORIZATION_ID" "$PHC2D_EXPECTED_COMMIT" "$job_id" > "$campaign/preflight_submission.env"
    ;;
  submit)
    [[ "$("$PHC2D_ENV/bin/python" -c "import json; print(json.load(open('$campaign/campaign_status.json'))['status'])")" == evaluation_plan_ready ]] || { echo "G3 validation preflight must pass before array submission" >&2; exit 5; }
    [[ -s "$campaign/preflight_manifest.csv" && -d "$campaign/test_cache" ]] || { echo "G3 preflight evidence/cache missing" >&2; exit 5; }
    [[ ! -e "$campaign/array_submission.env" ]] || { echo "G3 array already submitted; no retry is permitted" >&2; exit 5; }
    job_id="$(sbatch --parsable -p gpu_5090 --gres=gpu:1 -c 8 -t 08:00:00 --array=0-7%8 -J fwd020_g3 \
      -o "$campaign/slurm_%A_%a.out" -e "$campaign/slurm_%A_%a.err" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_ENV=$PHC2D_ENV,PHC2D_G3_CAMPAIGN=$campaign,PHC2D_G3_STAGE=shard" \
      "$PHC2D_ROOT/jobs/paracloud_exp_fwd_020_g3.sh)"
    printf 'authorization_id=%s\ndeployed_git_commit=%s\njob_id=%s\narray=0-7%%8\nstage=evaluation\n' "$PHC2D_FWD020_AUTHORIZATION_ID" "$PHC2D_EXPECTED_COMMIT" "$job_id" > "$campaign/array_submission.env"
    ;;
  status)
    "$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_bridge_campaign status-g3 --campaign-dir "$campaign" --repository-root "$PHC2D_ROOT"
    ;;
  finalize)
    "$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_bridge_campaign finalize-g3 --campaign-dir "$campaign" --repository-root "$PHC2D_ROOT"
    ;;
  *) echo "usage: $0 preflight|submit|status|finalize CAMPAIGN_DIR" >&2; exit 2 ;;
esac
