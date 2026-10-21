#!/usr/bin/env bash
# Fail-closed evaluation-only controller for EXP-FWD-023 on N56.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set the frozen deployment}"
: "${PHC2D_EXPECTED_COMMIT:?Set the frozen commit}"
: "${PHC2D_TRANSFER_AUTHORIZATION_ID:?Set the authorization identity}"
[[ "${PHC2D_AUTHORIZE_TRANSFER:-}" == YES ]] || { echo 'authorization missing' >&2; exit 4; }
cd "$PHC2D_ROOT"
[[ "$(tr -d '[:space:]' < DEPLOYED_GIT_COMMIT)" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'deployment mismatch' >&2; exit 3; }
PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1}"
export PATH="$PHC2D_ENV/bin:$PATH"
export PYTHONPATH="$PHC2D_ROOT/src"
command_name="${1:?usage: freeze|evaluation-batch|status|finalize campaign [batch]}"
campaign="${2:?Set the campaign path}"
case "$command_name" in
  freeze)
    python -m phc2d.forward.full_cell_transfer_campaign freeze \
      --config configs/research/full_cell_transfer_exp_fwd_023_v1.yaml \
      --output "$campaign" --repository-root "$PHC2D_ROOT"
    ;;
  evaluation-batch)
    batch="${3:?Set the batch index}"
    [[ "$batch" =~ ^[0-9]+$ ]] && (( batch <= 23 )) || { echo 'batch must be 0..23' >&2; exit 2; }
    marker="$campaign/evaluation/evaluation_batch_${batch}_submission.env"
    [[ ! -e "$marker" ]] || { echo 'batch already submitted; no retry permitted' >&2; exit 5; }
    python - "$campaign" "$PHC2D_EXPECTED_COMMIT" "$PHC2D_TRANSFER_AUTHORIZATION_ID" <<'PY'
import json, pathlib, sys, yaml
root = pathlib.Path(sys.argv[1])
profile = yaml.safe_load((root / 'config.resolved.yaml').read_text())
contract = profile['transfer_evaluation']
if contract['execution_start_git_commit'] != sys.argv[2] or contract['authorization_id'] != sys.argv[3]:
    raise SystemExit('frozen extension deployment/authorization mismatch')
result = pathlib.Path(contract['source_campaign']) / 'evaluation/results.json'
if json.loads(result.read_text()).get('status') != 'finalized_seed0_exploratory_surrogate_evidence':
    raise SystemExit('source six-domain evaluation must finalize first; no concurrent campaigns')
PY
    start=$((batch * 16)); end=$((start + 15)); (( end < 378 )) || end=377
    if (( batch > 0 )); then
      previous_start=$(((batch - 1) * 16)); previous_end=$((previous_start + 15))
      python -m phc2d.forward.full_cell_multidomain_campaign audit-evaluation \
        --campaign "$campaign" --start-index "$previous_start" --end-index "$previous_end"
    fi
    job_id="$(sbatch --parsable --array="$start-$end%7" -J "fwd023_eval_b${batch}" \
      -o "$campaign/slurm_eval_b${batch}_%A_%a.out" -e "$campaign/slurm_eval_b${batch}_%A_%a.err" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_ENV=$PHC2D_ENV,PHC2D_TRANSFER_CAMPAIGN=$campaign" \
      jobs/paracloud_exp_fwd_023_evaluate.sh)"
    printf 'job_id=%s\narray=%s-%s%%7\n' "$job_id" "$start" "$end" > "$marker"
    printf 'job_id=%s\n' "$job_id"
    ;;
  status)
    python -m phc2d.forward.full_cell_multidomain_campaign status --campaign "$campaign"
    ;;
  finalize)
    python -m phc2d.forward.full_cell_transfer_campaign finalize --campaign "$campaign"
    python -m phc2d.forward.full_cell_transfer_campaign render-report --campaign "$campaign"
    ;;
  *) echo "unknown command: $command_name" >&2; exit 2 ;;
esac
