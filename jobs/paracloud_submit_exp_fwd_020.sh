#!/usr/bin/env bash
# Freeze and submit EXP-FWD-020 stages under the N56 5090 no-retry contract.
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
stage="${1:?usage: $0 preflight|g1|g2|continuation-preflight|continuation-wave1|continuation-wave2}"; campaign="${2:-}"
mkdir -p "$PHC2D_RUN_ROOT"
if [[ "$stage" == preflight ]]; then
  campaign="$PHC2D_RUN_ROOT/$(date +%Y%m%d)/EXP-FWD-020_preflight_freeze_$(date +%Y%m%d%H%M%S)"
  "$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_bridge_campaign freeze-g1 --config configs/research/full_cell_global_local_bridge_exp_fwd_020_v1.yaml --output-dir "$campaign" --repository-root "$PHC2D_ROOT"
  preflight_run="$PHC2D_RUN_ROOT/$(date +%Y%m%d)/EXP-FWD-020_preflight_$(date +%Y%m%d%H%M%S)"
  job_id="$(sbatch --parsable -p gpu_5090 --gres=gpu:1 -c 8 -t 01:00:00 -J fwd020_preflight -o "$campaign/slurm_%j.out" -e "$campaign/slurm_%j.err" --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_FWD020_MANIFEST=$campaign/task_manifest.csv,PHC2D_FWD020_TASK_INDEX=0,PHC2D_PREFLIGHT_RUN_DIR=$preflight_run,PHC2D_ENV=$PHC2D_ENV" jobs/paracloud_exp_fwd_020_preflight.sh)"
  printf 'authorization_id=%s\ndeployed_git_commit=%s\nmanifest=%s\nmanifest_sha256=%s\npreflight_task_index=0\npreflight_run=%s\njob_id=%s\nstage=preflight\n' "$PHC2D_FWD020_AUTHORIZATION_ID" "$PHC2D_EXPECTED_COMMIT" "$campaign/task_manifest.csv" "$(sha256sum "$campaign/task_manifest.csv" | cut -d' ' -f1)" "$preflight_run" "$job_id" > "$campaign/submission.env"
  printf 'campaign=%s\njob_id=%s\n' "$campaign" "$job_id"; exit 0
elif [[ "$stage" == g1 ]]; then
  preflight_run="${2:?g1 requires the completed preflight run directory}"
  for artifact in run.log run.meta.json config.resolved.yaml metrics.json results.json history.csv model.pt; do test -s "$preflight_run/$artifact" || { echo "preflight evidence missing: $artifact" >&2; exit 5; }; done
  grep -q 'full_cell_forward_admission_probe_pass' "$preflight_run/results.json" || { echo "preflight did not pass" >&2; exit 5; }
  grep -q '2.7.1' "$preflight_run/run.meta.json" || { echo "required PyTorch 2.7.1 evidence missing" >&2; exit 5; }
  campaign="$PHC2D_RUN_ROOT/$(date +%Y%m%d)/EXP-FWD-020_g1_$(date +%Y%m%d%H%M%S)"
  "$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_bridge_campaign freeze-g1 --config configs/research/full_cell_global_local_bridge_exp_fwd_020_v1.yaml --output-dir "$campaign" --repository-root "$PHC2D_ROOT"
elif [[ "$stage" == continuation-preflight ]]; then
  g2_source="${2:?continuation-preflight requires the frozen 36-run G2 source campaign}"
  campaign="$PHC2D_RUN_ROOT/$(date +%Y%m%d)/EXP-FWD-020_epoch300_continuations_$(date +%Y%m%d%H%M%S)"
  "$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_bridge_campaign freeze-continuations --source-campaign "$g2_source" --output-dir "$campaign" --repository-root "$PHC2D_ROOT"
  probe_index="$($PHC2D_ENV/bin/python - "$campaign/task_manifest.csv" <<'PY'
import csv, sys
with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
row = next((item for item in rows if item["case_id"] == "f50_graded_candidate_s2"), None)
if row is None:
    raise SystemExit("required continuation representative is absent")
print(row["task_index"])
PY
)"
  probe_run="$campaign/admission/f50_graded_candidate_s2_epoch201"
  job_id="$(sbatch --parsable -p gpu_5090 --gres=gpu:1 -c 8 -t 02:00:00 -J fwd020_continue_probe -o "$campaign/slurm_admission_%j.out" -e "$campaign/slurm_admission_%j.err" --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_FWD020_MANIFEST=$campaign/task_manifest.csv,PHC2D_FWD020_TASK_INDEX=$probe_index,PHC2D_FWD020_PROBE_RUN=$probe_run,PHC2D_ENV=$PHC2D_ENV" jobs/paracloud_exp_fwd_020_continuation_preflight.sh)"
  printf 'authorization_id=%s\ndeployed_git_commit=%s\nmanifest=%s\nmanifest_sha256=%s\nrepresentative=f50_graded_candidate_s2\nprobe_run=%s\njob_id=%s\nstage=continuation_epoch201_admission\n' "$PHC2D_FWD020_AUTHORIZATION_ID" "$PHC2D_EXPECTED_COMMIT" "$campaign/task_manifest.csv" "$(sha256sum "$campaign/task_manifest.csv" | cut -d' ' -f1)" "$probe_run" "$job_id" > "$campaign/admission_submission.env"
  printf '%s\n' "$campaign" > "$PHC2D_RUN_ROOT/EXP-FWD-020_active.txt"
  printf 'campaign=%s\njob_id=%s\n' "$campaign" "$job_id"; exit 0
elif [[ "$stage" == continuation-wave1 || "$stage" == continuation-wave2 ]]; then
  campaign="${2:?continuation wave requires a frozen continuation campaign}"
  manifest="$campaign/task_manifest.csv"
  [[ -s "$manifest" && -s "$campaign/admission_submission.env" ]] || { echo "continuation freeze/admission submission missing" >&2; exit 5; }
  probe_run="$(sed -n 's/^probe_run=//p' "$campaign/admission_submission.env")"
  for artifact in run.log run.meta.json config.resolved.yaml history.csv metrics.json results.json model.pt model_summary.json; do test -s "$probe_run/$artifact" || { echo "continuation admission evidence missing: $artifact" >&2; exit 5; }; done
  grep -q 'full_cell_forward_admission_probe_pass' "$probe_run/results.json" || { echo "continuation admission did not pass" >&2; exit 5; }
  grep -q '"stop_epoch": 201' "$probe_run/results.json" || { echo "continuation admission did not execute epoch 201" >&2; exit 5; }
  if [[ "$stage" == continuation-wave1 ]]; then
    start=0; end=15; marker="$campaign/wave1_submission.env"
  else
    [[ -s "$campaign/wave1_submission.env" ]] || { echo "wave 1 was not submitted" >&2; exit 5; }
    "$PHC2D_ENV/bin/python" -m phc2d.forward.global_local_bridge_campaign audit-training-manifest --manifest "$manifest" --start-index 0 --end-index 15 --repository-root "$PHC2D_ROOT"
    start=16; end=28; marker="$campaign/wave2_submission.env"
  fi
  [[ ! -e "$marker" ]] || { echo "continuation wave already submitted; no retry is permitted" >&2; exit 5; }
  job_id="$(sbatch --parsable -p gpu_5090 --gres=gpu:1 -c 8 -t 24:00:00 --array="$start-$end%8" -J "fwd020_${stage}" -o "$campaign/slurm_%A_%a.out" -e "$campaign/slurm_%A_%a.err" --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_FWD020_MANIFEST=$manifest,PHC2D_ENV=$PHC2D_ENV" jobs/paracloud_exp_fwd_020_train.sh)"
  printf 'authorization_id=%s\ndeployed_git_commit=%s\nmanifest=%s\nmanifest_sha256=%s\narray=%s-%s%%8\njob_id=%s\nstage=%s\n' "$PHC2D_FWD020_AUTHORIZATION_ID" "$PHC2D_EXPECTED_COMMIT" "$manifest" "$(sha256sum "$manifest" | cut -d' ' -f1)" "$start" "$end" "$job_id" "$stage" > "$marker"
  printf 'campaign=%s\njob_id=%s\narray=%s-%s%%8\n' "$campaign" "$job_id" "$start" "$end"; exit 0
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
job_id="$(sbatch --parsable -p gpu_5090 --gres=gpu:1 -c 8 -t 24:00:00 --array="0-$((count - 1))%8" -J "fwd020_${stage}" -o "$campaign/slurm_%A_%a.out" -e "$campaign/slurm_%A_%a.err" --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_FWD020_MANIFEST=$manifest,PHC2D_ENV=$PHC2D_ENV" jobs/paracloud_exp_fwd_020_train.sh)"
printf 'authorization_id=%s\ndeployed_git_commit=%s\nmanifest=%s\nmanifest_sha256=%s\narray_concurrency=8\njob_id=%s\nstage=%s\n' "$PHC2D_FWD020_AUTHORIZATION_ID" "$PHC2D_EXPECTED_COMMIT" "$manifest" "$(sha256sum "$manifest" | cut -d' ' -f1)" "$job_id" "$stage" > "$campaign/submission.env"
printf '%s\n' "$campaign" > "$PHC2D_RUN_ROOT/EXP-FWD-020_active.txt"
printf 'campaign=%s\njob_id=%s\narray=0-%s%%8\n' "$campaign" "$job_id" "$((count - 1))"
