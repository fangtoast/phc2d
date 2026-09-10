#!/usr/bin/env bash
# Freeze, preflight, dry-run, or explicitly submit EXP-FWD-008 six-domain evaluation.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
mode="${1:-}"; authorization_id="${2:-}"
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"; PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"; PY="${PHC2D_PYTHON:-$PHC2D_ENV/bin/python}"
campaign="${PHC2D_FWD008_SIX_DOMAIN_CAMPAIGN:?Set PHC2D_FWD008_SIX_DOMAIN_CAMPAIGN}"
config="${PHC2D_FWD008_SIX_DOMAIN_CONFIG:-configs/research/full_cell_k5_six_domain_exp_fwd_008_n40_v1.yaml}"
cd "$PHC2D_ROOT"; campaign="$(realpath -m "$campaign")"
require_authorization() { [[ "${PHC2D_AUTHORIZE_EXP_FWD_008:-}" == "YES" && -n "$authorization_id" ]] || { echo "explicit EXP-FWD-008 authorization required" >&2; exit 4; }; }
case "$mode" in
  freeze) PYTHONPATH=src "$PY" -m phc2d.forward.full_cell_six_domain_n40 freeze --campaign-dir "$campaign" --repository-root "$PHC2D_ROOT" --config "$config" ;;
  preflight)
    require_authorization
    [[ ! -f "$campaign/preflight_submission.env" ]] || { echo "preflight already submitted; no retry" >&2; exit 6; }
    set +e; output="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 04:00:00 -J fwd008_preflight -o "$campaign/slurm_preflight_%j.out" -e "$campaign/slurm_preflight_%j.err" --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_FWD008_SIX_DOMAIN_CAMPAIGN=$campaign" jobs/paracloud_full_cell_six_domain_preflight.sh 2>&1)"; code=$?; set -e
    if [[ "$code" -ne 0 ]]; then printf 'failed_at=%s\nsbatch_output=%s\nnext_action=human_diagnosis_and_explicit_resubmission\n' "$(date -Is)" "$output" > "$campaign/preflight_submission_failure.txt"; echo "preflight submission failed; no automatic retry" >&2; exit "$code"; fi
    printf 'authorization_id=%s\npreflight_job_id=%s\nsubmitted_at=%s\n' "$authorization_id" "$output" "$(date -Is)" > "$campaign/preflight_submission.env" ;;
  dry-run) echo "campaign_dir=$campaign sbatch --array=0-9%8 --gres=gpu:1 -c 6 -t 04:00:00" ;;
  submit)
    require_authorization
    [[ ! -f "$campaign/submission.env" ]] || { echo "evaluation already submitted; no retry" >&2; exit 6; }
    [[ "$($PY -c "import json; print(json.load(open('$campaign/campaign_status.json'))['status'])")" == "evaluation_plan_ready" ]] || { echo "GPU preflight must pass before evaluation submission" >&2; exit 5; }
    set +e; output="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 04:00:00 --array=0-9%8 -J fwd008_sixdom -o "$campaign/slurm_shard_%A_%a.out" -e "$campaign/slurm_shard_%A_%a.err" --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_FWD008_SIX_DOMAIN_CAMPAIGN=$campaign" jobs/paracloud_full_cell_six_domain_exp_fwd_008.sh 2>&1)"; code=$?; set -e
    if [[ "$code" -ne 0 ]]; then printf 'failed_at=%s\nsbatch_output=%s\nnext_action=human_diagnosis_and_explicit_resubmission\n' "$(date -Is)" "$output" > "$campaign/submission_failure.txt"; echo "evaluation submission failed; no automatic retry" >&2; exit "$code"; fi
    printf 'authorization_id=%s\nshard_job_id=%s\nsubmitted_at=%s\n' "$authorization_id" "$output" "$(date -Is)" > "$campaign/submission.env" ;;
  *) echo "usage: $0 freeze|preflight|dry-run|submit [AUTHORIZATION_ID]" >&2; exit 2 ;;
esac
