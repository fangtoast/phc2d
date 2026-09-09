#!/usr/bin/env bash
# Freeze and submit EXP-FWD-020 stages under the N40 4090 no-retry contract.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d-exp-fwd-020}"; cd "$PHC2D_ROOT"
: "${PHC2D_EXPECTED_COMMIT:?Set the frozen full commit.}"
: "${PHC2D_FWD020_AUTHORIZATION_ID:?Set authorization identity.}"
: "${PHC2D_ENV:?Set the versioned runtime environment directory.}"
: "${PHC2D_RUN_ROOT:?Set the external EXP-FWD-020 evidence root.}"
export PYTHONPATH="$PHC2D_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
[[ "${PHC2D_AUTHORIZE_EXP_FWD_020:-}" == YES ]] || { echo "authorization missing" >&2; exit 4; }
[[ "$(tr -d '[:space:]' < DEPLOYED_GIT_COMMIT)" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo "deployment mismatch" >&2; exit 3; }
stage="${1:?usage: $0 preflight|g1|g2 [G1_CAMPAIGN_DIR]}"; campaign="${2:-}"
mkdir -p "$PHC2D_RUN_ROOT"
if [[ "$stage" == preflight ]]; then
  campaign="$PHC2D_RUN_ROOT/$(date +%Y%m%d)/EXP-FWD-020_preflight_freeze_$(date +%Y%m%d%H%M%S)"
  "$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_bridge_campaign freeze-g1 --config configs/research/full_cell_global_local_bridge_exp_fwd_020_v1.yaml --output-dir "$campaign" --repository-root "$PHC2D_ROOT"
  config="$(sed -n '2{s/\r$//;s/^[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\([^,]*\),.*/\1/p}' "$campaign/task_manifest.csv")"
  preflight_run="$PHC2D_RUN_ROOT/$(date +%Y%m%d)/EXP-FWD-020_preflight_$(date +%Y%m%d%H%M%S)"
  job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 01:00:00 -J fwd020_preflight -o "$campaign/slurm_%j.out" -e "$campaign/slurm_%j.err" --export="ALL,PHC2D_PREFLIGHT_CONFIG=$config,PHC2D_PREFLIGHT_RUN_DIR=$preflight_run,PHC2D_ENV=$PHC2D_ENV" jobs/paracloud_exp_fwd_020_preflight.sh)"
  printf 'authorization_id=%s\ndeployed_git_commit=%s\npreflight_config=%s\npreflight_run=%s\njob_id=%s\nstage=preflight\n' "$PHC2D_FWD020_AUTHORIZATION_ID" "$PHC2D_EXPECTED_COMMIT" "$config" "$preflight_run" "$job_id" > "$campaign/submission.env"
  printf 'campaign=%s\njob_id=%s\n' "$campaign" "$job_id"; exit 0
elif [[ "$stage" == g1 ]]; then
  preflight_run="${2:?g1 requires the completed preflight run directory}"
  for artifact in run.log run.meta.json config.resolved.yaml metrics.json results.json history.csv model.pt; do test -s "$preflight_run/$artifact" || { echo "preflight evidence missing: $artifact" >&2; exit 5; }; done
  grep -q 'full_cell_forward_admission_probe_pass' "$preflight_run/results.json" || { echo "preflight did not pass" >&2; exit 5; }
  grep -q '2.7.1' "$preflight_run/run.meta.json" || { echo "required PyTorch 2.7.1 evidence missing" >&2; exit 5; }
  campaign="$PHC2D_RUN_ROOT/$(date +%Y%m%d)/EXP-FWD-020_g1_$(date +%Y%m%d%H%M%S)"
  "$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_bridge_campaign freeze-g1 --config configs/research/full_cell_global_local_bridge_exp_fwd_020_v1.yaml --output-dir "$campaign" --repository-root "$PHC2D_ROOT"
elif [[ "$stage" == g2 && -n "$campaign" ]]; then
  g1="${3:?g2 requires G1 campaign directory}"
  next="$PHC2D_RUN_ROOT/$(date +%Y%m%d)/EXP-FWD-020_g2_$(date +%Y%m%d%H%M%S)"
  "$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_bridge_campaign freeze-g2 --config configs/research/full_cell_global_local_bridge_exp_fwd_020_v1.yaml --g1-dir "$g1" --output-dir "$next" --repository-root "$PHC2D_ROOT"; campaign="$next"
else
  echo "invalid stage invocation" >&2; exit 2
fi
manifest="$campaign/task_manifest.csv"; count="$(( $(wc -l < "$manifest") - 1 ))"
(( count > 0 )) || { echo "no G2 tasks passed G1; no submission"; exit 0; }
[[ ! -e "$campaign/submission.env" ]] || { echo "campaign already submitted" >&2; exit 5; }
job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 24:00:00 --array="0-$((count - 1))%8" -J "fwd020_${stage}" -o "$campaign/slurm_%A_%a.out" -e "$campaign/slurm_%A_%a.err" --export="ALL,PHC2D_FWD020_MANIFEST=$manifest,PHC2D_ENV=$PHC2D_ENV" jobs/paracloud_exp_fwd_020_train.sh)"
printf 'authorization_id=%s\ndeployed_git_commit=%s\nmanifest=%s\nmanifest_sha256=%s\narray_concurrency=8\njob_id=%s\nstage=%s\n' "$PHC2D_FWD020_AUTHORIZATION_ID" "$PHC2D_EXPECTED_COMMIT" "$manifest" "$(sha256sum "$manifest" | cut -d' ' -f1)" "$job_id" "$stage" > "$campaign/submission.env"
printf '%s\n' "$campaign" > "$PHC2D_RUN_ROOT/EXP-FWD-020_active.txt"
printf 'campaign=%s\njob_id=%s\narray=0-%s%%8\n' "$campaign" "$job_id" "$((count - 1))"
