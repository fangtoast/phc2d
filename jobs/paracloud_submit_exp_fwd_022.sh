#!/usr/bin/env bash
# Submit gated EXP-FWD-022 validation attribution stages on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

mode="${1:-}"
authorization_id="${2:-}"
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
: "${PHC2D_FWD022_CAMPAIGN:?Set PHC2D_FWD022_CAMPAIGN}"

[[ -n "$authorization_id" ]] || { echo "authorization id is required" >&2; exit 2; }
[[ -x "$PHC2D_ENV/bin/python" ]] || { echo "N40 Python runtime missing" >&2; exit 3; }
cd "$PHC2D_ROOT"
export PYTHONPATH="$PHC2D_ROOT/src"

status="$($PHC2D_ENV/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$PHC2D_FWD022_CAMPAIGN/campaign_status.json")"

case "$mode" in
  preflight)
    [[ "$status" == "frozen_preflight_required" ]] || { echo "preflight requires frozen_preflight_required, got $status" >&2; exit 4; }
    sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 01:00:00 \
      -J fwd022_preflight \
      -o "$PHC2D_FWD022_CAMPAIGN/slurm_preflight_%j.out" \
      -e "$PHC2D_FWD022_CAMPAIGN/slurm_preflight_%j.err" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_ENV=$PHC2D_ENV,PHC2D_FWD022_CAMPAIGN=$PHC2D_FWD022_CAMPAIGN,PHC2D_FWD022_STAGE=preflight,PHC2D_AUTHORIZATION_ID=$authorization_id" \
      jobs/paracloud_exp_fwd_022.sh
    ;;
  validation)
    [[ "$status" == "g0_preflight_passed" ]] || { echo "validation requires g0_preflight_passed, got $status" >&2; exit 5; }
    job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 12:00:00 --array=0-7%4 \
      -J fwd022_validation \
      -o "$PHC2D_FWD022_CAMPAIGN/slurm_shard_%A_%a.out" \
      -e "$PHC2D_FWD022_CAMPAIGN/slurm_shard_%A_%a.err" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_ENV=$PHC2D_ENV,PHC2D_FWD022_CAMPAIGN=$PHC2D_FWD022_CAMPAIGN,PHC2D_FWD022_STAGE=shard,PHC2D_AUTHORIZATION_ID=$authorization_id" \
      jobs/paracloud_exp_fwd_022.sh)"
    "$PHC2D_ENV/bin/python" - "$PHC2D_FWD022_CAMPAIGN/campaign_status.json" "$job_id" "$authorization_id" <<'PY'
import json, sys
path, job_id, authorization_id = sys.argv[1:]
value = json.load(open(path))
value.update(status="g1_submitted", validation_job_id=job_id, authorization_id=authorization_id)
open(path, "w").write(json.dumps(value, indent=2) + "\n")
PY
    sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 02:00:00 \
      --dependency="afterok:$job_id" -J fwd022_finalize \
      -o "$PHC2D_FWD022_CAMPAIGN/slurm_finalize_%j.out" \
      -e "$PHC2D_FWD022_CAMPAIGN/slurm_finalize_%j.err" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_ENV=$PHC2D_ENV,PHC2D_FWD022_CAMPAIGN=$PHC2D_FWD022_CAMPAIGN,PHC2D_FWD022_STAGE=finalize,PHC2D_AUTHORIZATION_ID=$authorization_id" \
      jobs/paracloud_exp_fwd_022.sh
    ;;
  status)
    "$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_branch_attribution status \
      --campaign-dir "$PHC2D_FWD022_CAMPAIGN" --repository-root "$PHC2D_ROOT"
    ;;
  *)
    echo "usage: $0 {preflight|validation|status} AUTHORIZATION_ID" >&2
    exit 2
    ;;
esac
