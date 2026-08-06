#!/usr/bin/env bash
# Plan or submit one full-cell inverse Stage-A wave as isolated N40 GPU jobs.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

mode="${1:-}"
config="${2:-}"
wave_id="${3:-}"
shift "$(( $# >= 3 ? 3 : $# ))"
variants=("$@")
if [[ "$mode" != "probe" && "$mode" != "train" ]]; then
  echo "usage: $0 probe|train CONFIG WAVE_ID VARIANT [VARIANT]" >&2
  exit 2
fi
if [[ ! -f "$config" || ! "$wave_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "CONFIG must exist and WAVE_ID must use only letters, digits, dot, underscore, or dash." >&2
  exit 2
fi
if (( ${#variants[@]} < 1 || ${#variants[@]} > 2 )); then
  echo "Select one or two variants per wave (9 or 18 jobs)." >&2
  exit 2
fi
if (( ${#variants[@]} == 2 )) && [[ "${variants[0]}" == "${variants[1]}" ]]; then
  echo "Wave variants must be unique." >&2
  exit 2
fi
for variant in "${variants[@]}"; do
  case "$variant" in
    binary_polar_holes_1|binary_polar_holes_2|binary_spectral_threshold|graded_rbf_fourier_1|graded_rbf_fourier_2) ;;
    *) echo "Unsupported variant: $variant" >&2; exit 2 ;;
  esac
done

PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
PHC2D_PYTHON="${PHC2D_PYTHON:-$PHC2D_ENV/bin/python}"
if [[ ! -x "$PHC2D_PYTHON" ]]; then
  echo "Missing Python interpreter: $PHC2D_PYTHON" >&2
  exit 2
fi
expected_stage="inverse_full_cell_stage_a_preflight"
job_script="jobs/paracloud_gpu_full_cell_inverse_stage_a_preflight.sh"
if [[ "$mode" == "train" ]]; then
  expected_stage="inverse_full_cell_stage_a_train"
  job_script="jobs/paracloud_gpu_full_cell_inverse_stage_a_train.sh"
fi
if [[ ! -f "$job_script" ]]; then
  echo "Missing job script: $job_script" >&2
  exit 2
fi

contract_rows_text="$(PYTHONPATH=src "$PHC2D_PYTHON" - "$config" "$expected_stage" <<'PY'
import sys

from phc2d.inverse.full_cell_training import (
    load_full_cell_stage_a_preflight_config,
    load_full_cell_stage_a_train_config,
)

loader = (
    load_full_cell_stage_a_preflight_config
    if sys.argv[2] == "inverse_full_cell_stage_a_preflight"
    else load_full_cell_stage_a_train_config
)
payload = loader(sys.argv[1])
actual_stage = payload.get("experiment", {}).get("stage")
if actual_stage != sys.argv[2]:
    raise SystemExit(f"config stage mismatch: {actual_stage!r} != {sys.argv[2]!r}")
datasets = payload.get("datasets", {})
if set(datasets) != {"binary", "graded"}:
    raise SystemExit("config must define exactly binary and graded datasets")
for route in ("binary", "graded"):
    dataset = datasets[route]
    print(route + "|" + dataset["path"] + "|" + dataset["expected_dataset_sha256"])
PY
)"
mapfile -t contract_rows <<< "$contract_rows_text"
for row in "${contract_rows[@]}"; do
  IFS='|' read -r route dataset_path expected_dataset_sha256 <<< "$row"
  if [[ ! -f "$dataset_path" ]]; then
    echo "Missing $route dataset: $dataset_path" >&2
    exit 3
  fi
  actual_dataset_sha256="$(sha256sum "$dataset_path" | awk '{print $1}')"
  if [[ "$actual_dataset_sha256" != "$expected_dataset_sha256" ]]; then
    echo "$route dataset SHA-256 mismatch: $actual_dataset_sha256 != $expected_dataset_sha256" >&2
    exit 3
  fi
done

methods=(direct_inverse_mlp single_mode_generator mixture_conditional_generator)
seeds=(0 1 2)
job_count=$(( ${#variants[@]} * ${#methods[@]} * ${#seeds[@]} ))
default_concurrency=9
if (( job_count == 18 )); then
  default_concurrency=18
fi
wave_concurrency="${PHC2D_WAVE_CONCURRENCY:-$default_concurrency}"
if [[ "$wave_concurrency" != "3" && "$wave_concurrency" != "9" && "$wave_concurrency" != "18" ]]; then
  echo "PHC2D_WAVE_CONCURRENCY must be 3, 9, or 18." >&2
  exit 2
fi
if (( wave_concurrency > job_count )); then
  echo "PHC2D_WAVE_CONCURRENCY=$wave_concurrency exceeds job_count=$job_count." >&2
  exit 2
fi
dry_run="${PHC2D_SUBMIT_DRY_RUN:-0}"
if [[ "$dry_run" != "0" && "$dry_run" != "1" ]]; then
  echo "PHC2D_SUBMIT_DRY_RUN must be 0 or 1." >&2
  exit 2
fi
if [[ "$dry_run" == "0" ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required unless PHC2D_SUBMIT_DRY_RUN=1." >&2
  exit 2
fi

timestamp="$(date +%Y%m%d%H%M%S)"
manifest_dir="runs/$(date +%Y%m%d)/${wave_id}_${mode}_${timestamp}"
manifest="$manifest_dir/submission_manifest.csv"
mkdir -p "$manifest_dir"
printf 'variant,method,seed,job_id,dependency,run_dir,mode,submit_state\n' > "$manifest"
printf 'mode=%s\nconfig=%s\nvariants=%s\njob_count=%s\nconcurrency=%s\ndry_run=%s\n' \
  "$mode" "$config" "${variants[*]}" "$job_count" "$wave_concurrency" "$dry_run" \
  > "$manifest_dir/submission.meta"

task_index=0
previous_group_ids=()
current_group_ids=()
for variant in "${variants[@]}"; do
  for method in "${methods[@]}"; do
    for seed in "${seeds[@]}"; do
      run_dir="runs/$(date +%Y%m%d)/${wave_id}_${mode}_${timestamp}_${variant}_${method}_seed${seed}"
      time_limit="00:10:00"
      if [[ "$mode" == "train" ]]; then
        time_limit="12:00:00"
      fi
      sbatch_args=(-p gpu_4090 --gres=gpu:1 -c 6 -t "$time_limit")
      dependency_value=""
      if (( task_index >= wave_concurrency )) && (( task_index % wave_concurrency == 0 )); then
        previous_group_ids=("${current_group_ids[@]}")
        current_group_ids=()
      fi
      if (( task_index >= wave_concurrency )); then
        dependency="$(IFS=:; echo "${previous_group_ids[*]}")"
        dependency_value="afterany:$dependency"
        sbatch_args+=(--dependency="$dependency_value")
      fi
      submit_state="submitted"
      if [[ "$dry_run" == "1" ]]; then
        job_id="DRYRUN-$(printf '%03d' "$((task_index + 1))")"
        submit_state="dry_run"
      elif [[ "$mode" == "probe" ]]; then
        job_id="$(sbatch --parsable "${sbatch_args[@]}" \
          -J "invp_${variant}_${method}_s${seed}" \
          -o "${run_dir}.slurm.out" -e "${run_dir}.slurm.err" \
          --export="ALL,PHC2D_PREFLIGHT_CONFIG=$config,PHC2D_PREFLIGHT_VARIANT=$variant,PHC2D_PREFLIGHT_METHOD=$method,PHC2D_PREFLIGHT_SEED=$seed,PHC2D_PREFLIGHT_RUN_DIR=$run_dir" \
          "$job_script")"
      else
        job_id="$(sbatch --parsable "${sbatch_args[@]}" \
          -J "invt_${variant}_${method}_s${seed}" \
          -o "${run_dir}.slurm.out" -e "${run_dir}.slurm.err" \
          --export="ALL,PHC2D_CONFIG=$config,PHC2D_VARIANT=$variant,PHC2D_METHOD=$method,PHC2D_SEED=$seed,PHC2D_RUN_DIR=$run_dir" \
          "$job_script")"
      fi
      current_group_ids+=("$job_id")
      task_index=$((task_index + 1))
      printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$variant" "$method" "$seed" "$job_id" "$dependency_value" "$run_dir" "$mode" \
        "$submit_state" >> "$manifest"
    done
  done
done
if [[ "$dry_run" == "0" ]]; then
  printf '%s\n' "$manifest" > runs/full_cell_inverse_stage_a_active_wave.txt
fi
echo "$manifest"
