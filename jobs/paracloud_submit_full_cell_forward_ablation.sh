#!/usr/bin/env bash
# Submit the frozen 3-configuration x 2-resolution x 3-seed ablation matrix.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

mode="${1:-}"
selection_csv="${2:-}"
campaign_id="${3:-}"
if [[ "$mode" != "probe" && "$mode" != "train" ]]; then
  echo "usage: $0 probe|train SELECTION_CSV CAMPAIGN_ID" >&2
  exit 2
fi
if [[ ! -f "$selection_csv" || -z "$campaign_id" ]]; then
  echo "SELECTION_CSV must exist and CAMPAIGN_ID must be non-empty" >&2
  exit 2
fi

PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
wave_concurrency="${PHC2D_WAVE_CONCURRENCY:-6}"
if [[ "$wave_concurrency" != "3" && "$wave_concurrency" != "6" ]]; then
  echo "ablation PHC2D_WAVE_CONCURRENCY must be 3 or 6" >&2
  exit 2
fi
selection_count="$(tail -n +2 "$selection_csv" | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')"
if [[ "$selection_count" != "3" ]]; then
  echo "selection CSV must contain exactly three frozen configurations" >&2
  exit 3
fi

timestamp="$(date +%Y%m%d%H%M%S)"
manifest="runs/$(date +%Y%m%d)/${campaign_id}_${mode}_${timestamp}/submission_manifest.csv"
mkdir -p "$(dirname "$manifest")"
printf 'lane,representation,config,model,threshold,resolution,seed,job_id,run_dir,mode\n' > "$manifest"
task_index=0
previous_group_ids=()
current_group_ids=()

while IFS=, read -r lane representation config model threshold; do
  if [[ ! -f "$config" ]]; then
    echo "missing frozen config: $config" >&2
    exit 3
  fi
  for resolution in 32 128; do
    for seed in 0 1 2; do
      run_dir="runs/$(date +%Y%m%d)/${campaign_id}_${mode}_${timestamp}_${lane}_${representation}_${model}_r${resolution}_seed${seed}"
      probe=0
      time_limit="12:00:00"
      if [[ "$mode" == "probe" ]]; then
        probe=1
        time_limit="00:15:00"
      fi
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
        -J "fc_${mode}_${model}_r${resolution}_s${seed}" \
        -o "${run_dir}.slurm.out" -e "${run_dir}.slurm.err" \
        --export="ALL,PHC2D_FORWARD_CONFIG=$config,PHC2D_FORWARD_MODEL=$model,PHC2D_FORWARD_SEED=$seed,PHC2D_FORWARD_RUN_DIR=$run_dir,PHC2D_FORWARD_ADMISSION_PROBE=$probe,PHC2D_FORWARD_THRESHOLD=$threshold,PHC2D_FORWARD_RESOLUTION=$resolution" \
        jobs/paracloud_full_cell_forward.sh)"
      current_group_ids+=("$job_id")
      task_index=$((task_index + 1))
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$lane" "$representation" "$config" "$model" "$threshold" "$resolution" "$seed" "$job_id" "$run_dir" "$mode" >> "$manifest"
    done
  done
done < <(tail -n +2 "$selection_csv")
if [[ "$task_index" != "18" ]]; then
  echo "internal error: expected 18 submitted tasks, got $task_index" >&2
  exit 4
fi
printf '%s\n' "$manifest" > runs/full_cell_forward_active_wave.txt
echo "$manifest"
