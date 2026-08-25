#!/usr/bin/env bash
# Freeze, preflight, or submit EXP-FWD-009 six-domain evaluation.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
mode="${1:-}"
authorization_id="${2:-}"
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
PY="${PHC2D_PYTHON:-$PHC2D_ENV/bin/python}"
campaign="${PHC2D_FWD009_SIX_DOMAIN_CAMPAIGN:?Set PHC2D_FWD009_SIX_DOMAIN_CAMPAIGN}"
source_campaign="${PHC2D_FWD008_SOURCE_CAMPAIGN:?Set PHC2D_FWD008_SOURCE_CAMPAIGN}"
continuation_campaign="${PHC2D_FWD009_CONTINUATION_CAMPAIGN:?Set PHC2D_FWD009_CONTINUATION_CAMPAIGN}"
config="${PHC2D_FWD009_SIX_DOMAIN_CONFIG:-configs/research/full_cell_k5_six_domain_exp_fwd_009_n40_v1.yaml}"
inventory="$continuation_campaign/evaluation_checkpoint_inventory.csv"
cd "$PHC2D_ROOT"
campaign="$(realpath -m "$campaign")"
require_authorization() {
  [[ "${PHC2D_AUTHORIZE_EXP_FWD_009:-}" == "YES" && -n "$authorization_id" ]] || {
    echo "explicit EXP-FWD-009 authorization required" >&2
    exit 4
  }
}
case "$mode" in
  freeze)
    PYTHONPATH=src "$PY" -m phc2d.forward.full_cell_continuation_campaign finalize-evaluation-inventory \
      --source-campaign "$source_campaign" --continuation-campaign "$continuation_campaign"
    PYTHONPATH=src "$PY" -m phc2d.forward.full_cell_six_domain_n40 freeze \
      --campaign-dir "$campaign" --repository-root "$PHC2D_ROOT" --config "$config" --k5-inventory "$inventory"
    ;;
  preflight)
    require_authorization
    [[ ! -f "$campaign/preflight_submission.env" ]] || { echo "preflight already submitted; no retry" >&2; exit 6; }
    set +e
    output="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 04:00:00 -J fwd009_preflight \
      -o "$campaign/slurm_preflight_%j.out" -e "$campaign/slurm_preflight_%j.err" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_SIX_DOMAIN_CAMPAIGN=$campaign" \
      jobs/paracloud_full_cell_six_domain_preflight.sh 2>&1)"
    code=$?
    set -e
    if [[ "$code" -ne 0 ]]; then
      printf 'failed_at=%s\nsbatch_output=%s\nnext_action=human_diagnosis_and_explicit_resubmission\n' "$(date -Is)" "$output" > "$campaign/preflight_submission_failure.txt"
      echo "preflight submission failed; no automatic retry" >&2
      exit "$code"
    fi
    printf 'authorization_id=%s\npreflight_job_id=%s\nsubmitted_at=%s\n' "$authorization_id" "$output" "$(date -Is)" > "$campaign/preflight_submission.env"
    echo "preflight_job_id=$output"
    ;;
  submit)
    require_authorization
    [[ ! -f "$campaign/submission.env" ]] || { echo "evaluation already submitted; no retry" >&2; exit 6; }
    [[ "$("$PY" -c "import json; print(json.load(open('$campaign/campaign_status.json'))['status'])")" == "evaluation_plan_ready" ]] || { echo "GPU preflight must pass before evaluation submission" >&2; exit 5; }
    shard_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 04:00:00 --array=0-9%8 -J fwd009_sixdom \
      -o "$campaign/slurm_shard_%A_%a.out" -e "$campaign/slurm_shard_%A_%a.err" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_FWD009_SIX_DOMAIN_CAMPAIGN=$campaign" \
      jobs/paracloud_full_cell_six_domain_exp_fwd_009.sh)"
    finalizer_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 02:00:00 --dependency="afterok:$shard_id" -J fwd009_finalize \
      -o "$campaign/slurm_finalizer_%j.out" -e "$campaign/slurm_finalizer_%j.err" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_SIX_DOMAIN_CAMPAIGN=$campaign" \
      jobs/paracloud_full_cell_six_domain_finalize.sh)"
    printf 'authorization_id=%s\nshard_job_id=%s\nfinalizer_job_id=%s\nsubmitted_at=%s\n' "$authorization_id" "$shard_id" "$finalizer_id" "$(date -Is)" > "$campaign/submission.env"
    echo "shard_job_id=$shard_id finalizer_job_id=$finalizer_id"
    ;;
  *) echo "usage: $0 freeze|preflight|submit AUTHORIZATION_ID" >&2; exit 2 ;;
esac
