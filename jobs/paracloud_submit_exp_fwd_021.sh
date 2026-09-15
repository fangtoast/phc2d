#!/usr/bin/env bash
# Manual, fail-closed controller for EXP-FWD-021 on N56 hp_5090.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d-exp-fwd-021}"
PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1}"
PHC2D_RUN_ROOT="${PHC2D_RUN_ROOT:-/data/run01/scxl892/runs/EXP-FWD-021}"
: "${PHC2D_EXPECTED_COMMIT:?Set the frozen deployed commit.}"
: "${PHC2D_FWD021_AUTHORIZATION_ID:?Set the authorization identity.}"
[[ "${PHC2D_AUTHORIZE_EXP_FWD_021:-}" == YES ]] || { echo "authorization missing" >&2; exit 4; }
cd "$PHC2D_ROOT"
[[ "$(tr -d '[:space:]' < DEPLOYED_GIT_COMMIT)" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo "deployment mismatch" >&2; exit 3; }
export PATH="$PHC2D_ENV/bin:$PATH"
export PYTHONPATH="$PHC2D_ROOT/src"

command_name="${1:?usage: $0 freeze|admission|audit-admission|train-batch|freeze-evaluation|evaluation-batch|status|finalize [campaign] [batch]}"
campaign="${2:-}"
case "$command_name" in
  freeze)
    mkdir -p "$PHC2D_RUN_ROOT/$(date +%Y%m%d)"
    campaign="$PHC2D_RUN_ROOT/$(date +%Y%m%d)/EXP-FWD-021_$(date +%Y%m%d%H%M%S)"
    python -m phc2d.forward.full_cell_multidomain_campaign freeze \
      --config configs/research/full_cell_multidomain_exp_fwd_021_v1.yaml \
      --output "$campaign" --repository-root "$PHC2D_ROOT" --dataset-path-key n56_path
    printf '%s\n' "$campaign" > "$PHC2D_RUN_ROOT/EXP-FWD-021_active.txt"
    printf 'authorization_id=%s\ndeployed_git_commit=%s\ncampaign=%s\n' \
      "$PHC2D_FWD021_AUTHORIZATION_ID" "$PHC2D_EXPECTED_COMMIT" "$campaign" > "$campaign/authorization.env"
    printf 'campaign=%s\n' "$campaign"
    ;;
  admission)
    : "${campaign:?admission requires the frozen campaign}"
    [[ ! -e "$campaign/admission_submission.env" ]] || { echo "admission already submitted; no retry permitted" >&2; exit 5; }
    job_id="$(sbatch --parsable --array=54-62%7 -J fwd021_admission \
      -o "$campaign/slurm_admission_%A_%a.out" -e "$campaign/slurm_admission_%A_%a.err" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_ENV=$PHC2D_ENV,PHC2D_FWD021_CAMPAIGN=$campaign,PHC2D_FWD021_ADMISSION=1" \
      jobs/paracloud_exp_fwd_021_train.sh)"
    printf 'job_id=%s\narray=54-62%%7\nmodels=9\nregime=S3-BG\n' "$job_id" > "$campaign/admission_submission.env"
    printf 'job_id=%s\n' "$job_id"
    ;;
  audit-admission)
    : "${campaign:?audit-admission requires the frozen campaign}"
    for index in $(seq 54 62); do
      model="$(python - "$campaign/task_manifest.csv" "$index" <<'PY'
import csv, sys
with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    print(next(row["model"] for row in csv.DictReader(handle) if int(row["task_index"]) == int(sys.argv[2])))
PY
)"
      run="$campaign/admission/${model}_s0"
      for artifact in run.log run.meta.json config.resolved.yaml history.csv metrics.json results.json model.pt admission_epoch1.pt model_summary.json; do
        test -s "$run/$artifact" || { echo "admission evidence missing: $model/$artifact" >&2; exit 5; }
      done
      grep -q 'multidomain_admission_probe_pass' "$run/results.json" || { echo "admission did not pass: $model" >&2; exit 5; }
      grep -q 'RTX 5090' "$run/metrics.json" || { echo "RTX 5090 device evidence missing: $model" >&2; exit 5; }
    done
    printf 'status=passed\nmodels=9\n' > "$campaign/admission_audit.env"
    ;;
  train-batch)
    : "${campaign:?train-batch requires the frozen campaign}"
    batch="${3:?train-batch requires batch 0 through 4}"
    [[ -s "$campaign/admission_audit.env" ]] || { echo "admission audit is not passed" >&2; exit 5; }
    starts=(0 16 18 34 36 52); ends=(15 17 33 35 51 62)
    (( batch >= 0 && batch < 6 )) || { echo "training batch must be 0 through 5" >&2; exit 2; }
    start="${starts[$batch]}"; end="${ends[$batch]}"
    marker="$campaign/training_batch_${batch}_submission.env"
    [[ ! -e "$marker" ]] || { echo "training batch already submitted; no retry permitted" >&2; exit 5; }
    if (( batch > 0 )); then
      previous=$((batch - 1)); previous_start="${starts[$previous]}"; previous_end="${ends[$previous]}"
      python -m phc2d.forward.full_cell_multidomain_campaign audit-training \
        --campaign "$campaign" --start-index "$previous_start" --end-index "$previous_end"
    fi
    job_id="$(sbatch --parsable --array="$start-$end%7" -J "fwd021_train_b${batch}" \
      -o "$campaign/slurm_train_b${batch}_%A_%a.out" -e "$campaign/slurm_train_b${batch}_%A_%a.err" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_ENV=$PHC2D_ENV,PHC2D_FWD021_CAMPAIGN=$campaign" \
      jobs/paracloud_exp_fwd_021_train.sh)"
    printf 'job_id=%s\narray=%s-%s%%7\n' "$job_id" "$start" "$end" > "$marker"
    printf 'job_id=%s\n' "$job_id"
    ;;
  freeze-evaluation)
    : "${campaign:?freeze-evaluation requires the frozen campaign}"
    python -m phc2d.forward.full_cell_multidomain_campaign audit-training --campaign "$campaign" --start-index 0 --end-index 62
    python -m phc2d.forward.full_cell_multidomain_campaign freeze-evaluation --campaign "$campaign"
    ;;
  evaluation-batch)
    : "${campaign:?evaluation-batch requires the frozen campaign}"
    batch="${3:?evaluation-batch requires batch 0 through 23}"
    (( batch >= 0 && batch < 24 )) || { echo "evaluation batch must be 0 through 23" >&2; exit 2; }
    start=$((batch * 16)); end=$((start + 15)); (( end > 377 )) && end=377
    marker="$campaign/evaluation/evaluation_batch_${batch}_submission.env"
    [[ ! -e "$marker" ]] || { echo "evaluation batch already submitted; no retry permitted" >&2; exit 5; }
    if (( batch > 0 )); then
      previous_start=$(((batch - 1) * 16)); previous_end=$((previous_start + 15))
      python - "$campaign" "$previous_start" "$previous_end" <<'PY'
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1]); start, end = int(sys.argv[2]), int(sys.argv[3])
with (root / "evaluation/evaluation_manifest.csv").open(encoding="utf-8", newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if start <= int(row["cell_index"]) <= end]
if len(rows) != end - start + 1:
    raise SystemExit("previous evaluation batch manifest is incomplete")
for row in rows:
    result = root / row["cell_dir"] / "results.json"
    if not result.is_file() or json.loads(result.read_text())["status"] != "evaluation_cell_success":
        raise SystemExit(f"previous evaluation batch is incomplete: {row['cell_id']}")
PY
    fi
    job_id="$(sbatch --parsable --array="$start-$end%7" -J "fwd021_eval_b${batch}" \
      -o "$campaign/slurm_eval_b${batch}_%A_%a.out" -e "$campaign/slurm_eval_b${batch}_%A_%a.err" \
      --export="ALL,PHC2D_ROOT=$PHC2D_ROOT,PHC2D_ENV=$PHC2D_ENV,PHC2D_FWD021_CAMPAIGN=$campaign" \
      jobs/paracloud_exp_fwd_021_evaluate.sh)"
    printf 'job_id=%s\narray=%s-%s%%7\n' "$job_id" "$start" "$end" > "$marker"
    printf 'job_id=%s\n' "$job_id"
    ;;
  status)
    : "${campaign:?status requires the frozen campaign}"
    python -m phc2d.forward.full_cell_multidomain_campaign status --campaign "$campaign"
    ;;
  finalize)
    : "${campaign:?finalize requires the frozen campaign}"
    python -m phc2d.forward.full_cell_multidomain_campaign finalize --campaign "$campaign"
    ;;
  *) echo "unknown command: $command_name" >&2; exit 2 ;;
esac
