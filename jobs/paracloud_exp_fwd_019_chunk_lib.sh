#!/usr/bin/env bash
# Shared bounded-array submission contract for EXP-FWD-019 on N40.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

phc2d_fwd019_submit_chunk() {
  local screen="$1" start="$2" end="$3" expected_count="$4" time_limit="$5" job_name="$6" scope="$7"
  local manifest="$screen/task_manifest.csv" count evidence job_id
  count="$(($(wc -l < "$manifest") - 1))"
  [[ "$count" == "$expected_count" ]] || { echo "unexpected task count: $count" >&2; return 5; }
  [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ ]] || { echo "chunk bounds must be integers" >&2; return 6; }
  (( start >= 0 && start <= end && end < count && end - start + 1 <= 16 )) || {
    echo "chunk must contain at most 16 tasks within 0-$((count - 1))" >&2; return 6;
  }

  shopt -s nullglob
  local prior=("$screen"/submission_chunk_*.env)
  if (( ${#prior[@]} == 0 )); then
    (( start == 0 )) || { echo "the first chunk must start at task 0" >&2; return 7; }
  else
    local highest_end=-1 prior_start prior_end prior_job states prior_evidence
    for prior_evidence in "${prior[@]}"; do
      prior_start="$(sed -n 's/^array_start=//p' "$prior_evidence")"
      prior_end="$(sed -n 's/^array_end=//p' "$prior_evidence")"
      prior_job="$(sed -n 's/^job_id=//p' "$prior_evidence")"
      (( start > prior_end || end < prior_start )) || { echo "chunk overlaps $prior_evidence" >&2; return 7; }
      (( prior_end > highest_end )) && highest_end="$prior_end"
      states="$(sacct -j "$prior_job" -X -n -P -o State | sed '/^[[:space:]]*$/d' | sort -u)"
      [[ "$states" == "COMPLETED" ]] || { echo "prior chunk $prior_job is not completely successful: $states" >&2; return 8; }
    done
    (( start == highest_end + 1 )) || { echo "chunks must be submitted contiguously" >&2; return 7; }
  fi

  evidence="$screen/submission_chunk_$(printf '%03d' "$start")_$(printf '%03d' "$end").env"
  [[ ! -e "$evidence" ]] || { echo "chunk submission evidence already exists" >&2; return 9; }
  job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t "$time_limit" --array="${start}-${end}%3" \
    -J "$job_name" -o "$screen/slurm_%A_%a.out" -e "$screen/slurm_%A_%a.err" \
    --export="ALL,PHC2D_FWD019_SCREEN_MANIFEST=$manifest" jobs/paracloud_exp_fwd_019_seed0_train.sh)"
  printf 'authorization_id=%s\ndeployed_git_commit=%s\nmanifest=%s\nmanifest_sha256=%s\narray_start=%s\narray_end=%s\njob_id=%s\nsubmission_scope=%s\n' \
    "$PHC2D_EXPECTED_FWD019_AUTHORIZATION_ID" "$PHC2D_EXPECTED_COMMIT" "$manifest" "$(sha256sum "$manifest" | cut -d' ' -f1)" \
    "$start" "$end" "$job_id" "$scope" > "$evidence"
  PHC2D_FWD019_SUBMITTED_JOB_ID="$job_id"
  PHC2D_FWD019_SUBMISSION_EVIDENCE="$evidence"
}
