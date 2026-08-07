#!/usr/bin/env bash
# Submit one 4-model x 3-seed full-cell forward wave as isolated GPU jobs.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

mode="${1:-}"
config="${2:-}"
wave_id="${3:-}"
threshold="${4:-}"
resolution="${5:-}"
train_size="${6:-}"
max_epochs="${7:-}"
if [[ "$mode" != "probe" && "$mode" != "train" ]]; then
  echo "usage: $0 probe|train CONFIG WAVE_ID [THRESHOLD] [RESOLUTION] [TRAIN_SIZE] [MAX_EPOCHS]" >&2
  exit 2
fi
if [[ "$mode" == "train" && ( -n "$train_size" || -n "$max_epochs" ) ]]; then
  if [[ -z "$train_size" || -z "$max_epochs" ]]; then
    echo "TRAIN_SIZE and MAX_EPOCHS must be supplied together" >&2
    exit 2
  fi
  if [[ "$train_size" != "8000" && "$train_size" != "20000" && "$train_size" != "40000" ]]; then
    echo "formal 50k TRAIN_SIZE must be 8000, 20000, or 40000" >&2
    exit 2
  fi
  if [[ "$max_epochs" != "20" && "$max_epochs" != "100" && "$max_epochs" != "200" ]]; then
    echo "formal 50k MAX_EPOCHS must be 20, 100, or 200" >&2
    exit 2
  fi
fi
if [[ ! -f "$config" || -z "$wave_id" ]]; then
  echo "CONFIG must exist and WAVE_ID must be non-empty" >&2
  exit 2
fi

PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
if [[ ! -x "$PHC2D_ENV/bin/python" ]]; then
  echo "missing GPU Python environment: $PHC2D_ENV" >&2
  exit 2
fi
contract="$($PHC2D_ENV/bin/python - "$config" <<'PY'
import sys, yaml
payload = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
dataset = payload["dataset"]
print(dataset["path"] + "|" + dataset["expected_dataset_sha256"])
PY
)"
dataset_path="${contract%%|*}"
expected_dataset_sha256="${contract#*|}"
actual_dataset_sha256="$(sha256sum "$dataset_path" | awk '{print $1}')"
if [[ "$actual_dataset_sha256" != "$expected_dataset_sha256" ]]; then
  echo "dataset SHA-256 mismatch: $actual_dataset_sha256 != $expected_dataset_sha256" >&2
  exit 3
fi

models=(mlp res_mlp_wide lifted_cnn fno2d_band_v2)
seeds=(0 1 2)
wave_concurrency="${PHC2D_WAVE_CONCURRENCY:-12}"
if [[ "$wave_concurrency" != "3" && "$wave_concurrency" != "6" && "$wave_concurrency" != "12" ]]; then
  echo "PHC2D_WAVE_CONCURRENCY must be 3, 6, or 12" >&2
  exit 2
fi
timestamp="$(date +%Y%m%d%H%M%S)"
manifest="runs/$(date +%Y%m%d)/${wave_id}_${mode}_${timestamp}/submission_manifest.csv"
mkdir -p "$(dirname "$manifest")"
printf 'model,seed,job_id,run_dir,mode\n' > "$manifest"
task_index=0
previous_group_ids=()
current_group_ids=()
for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    run_dir="runs/$(date +%Y%m%d)/${wave_id}_${mode}_${timestamp}_${model}_seed${seed}"
    probe=0
    time_limit="${PHC2D_FORWARD_TIME_LIMIT:-12:00:00}"
    if [[ "$mode" == "probe" ]]; then
      probe=1
      time_limit="00:10:00"
    fi
    # N40 assigns 60 GB host memory per requested RTX4090 by policy.  An
    # explicit 60000M request is interpreted as exceeding that per-card limit.
    sbatch_args=(-p gpu_4090 --gres=gpu:1 -c 6 -t "$time_limit")
    if (( task_index >= wave_concurrency )) && (( task_index % wave_concurrency == 0 )); then
      previous_group_ids=("${current_group_ids[@]}")
      current_group_ids=()
    fi
    if (( task_index >= wave_concurrency )); then
      dependency="$(IFS=:; echo "${previous_group_ids[*]}")"
      sbatch_args+=(--dependency="afterany:$dependency")
    fi
    job_id="$(sbatch --parsable "${sbatch_args[@]}" \
      -J "fc_${mode}_${model}_s${seed}" \
      -o "${run_dir}.slurm.out" -e "${run_dir}.slurm.err" \
      --export="ALL,PHC2D_FORWARD_CONFIG=$config,PHC2D_FORWARD_MODEL=$model,PHC2D_FORWARD_SEED=$seed,PHC2D_FORWARD_RUN_DIR=$run_dir,PHC2D_FORWARD_ADMISSION_PROBE=$probe,PHC2D_FORWARD_THRESHOLD=$threshold,PHC2D_FORWARD_RESOLUTION=$resolution,PHC2D_FORWARD_TRAIN_SIZE=$train_size,PHC2D_FORWARD_MAX_EPOCHS=$max_epochs" \
      jobs/paracloud_full_cell_forward.sh)"
    current_group_ids+=("$job_id")
    task_index=$((task_index + 1))
    printf '%s,%s,%s,%s,%s\n' "$model" "$seed" "$job_id" "$run_dir" "$mode" >> "$manifest"
  done
done
printf '%s\n' "$manifest" > runs/full_cell_forward_active_wave.txt
echo "$manifest"
