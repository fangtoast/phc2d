#!/usr/bin/env bash
# Submit a source-sharded paired inverse campaign with a unique receipt.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment root}"
: "${PHC2D_GPU_PLATFORM:?Set n40 or n56}"
: "${PHC2D_MANIFEST:?Set frozen campaign manifest}"
: "${PHC2D_MANIFEST_SHA256:?Set frozen manifest hash}"
: "${PHC2D_SOURCE_DIRECTORY:?Set frozen source directory}"
: "${PHC2D_CHECKPOINT:?Set frozen model}"
: "${PHC2D_METHOD:?Set adam, de or random}"
: "${PHC2D_OUTPUT_ROOT:?Set unique output directory}"
: "${PHC2D_RECEIPT_ROOT:?Set unique receipt directory}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployed code identity}"
case "$PHC2D_GPU_PLATFORM:$PHC2D_METHOD" in
  n40:adam|n56:de|n56:random|n40:feedback_seed|n56:feedback_seed|n40:feedback_round|n56:feedback_round) ;;
  n56:adam) [[ "${PHC2D_N56_TAKEOVER:-0}" == 1 ]] || { echo 'explicit N56 takeover required' >&2; exit 2; } ;;
  *) echo 'unsupported method/platform assignment' >&2; exit 2 ;;
esac
[[ "$(tr -d '[:space:]' < "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT")" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'deployment mismatch' >&2; exit 2; }
printf '%s  %s\n' "$PHC2D_MANIFEST_SHA256" "$PHC2D_MANIFEST" | sha256sum -c - >/dev/null
[[ -f "$PHC2D_CHECKPOINT" && -d "$PHC2D_SOURCE_DIRECTORY" ]] || { echo 'missing inputs' >&2; exit 2; }
worker="${PHC2D_JOB_SCRIPT:-$PHC2D_ROOT/jobs/paracloud_inverse_differentiable_campaign.sh}"
[[ -f "$worker" ]] || { echo 'missing worker script' >&2; exit 2; }
if [[ -n "${PHC2D_JOB_SCRIPT:-}" && -z "${PHC2D_WORKER_SHA256:-}" ]]; then
  echo 'overridden worker requires explicit hash' >&2; exit 2
fi
if [[ -n "${PHC2D_WORKER_SHA256:-}" ]]; then
  printf '%s  %s\n' "$PHC2D_WORKER_SHA256" "$worker" | sha256sum -c - >/dev/null
fi
if [[ "${PHC2D_SUBMIT_DRY_RUN:-0}" == 1 ]]; then
  printf 'prepared platform=%s method=%s sources=3 cpus=6 gpus=1 wall_minutes=60\n' "$PHC2D_GPU_PLATFORM" "$PHC2D_METHOD"
  exit 0
fi
[[ "${PHC2D_SUBMIT_DRY_RUN:-0}" == 0 ]] || { echo 'invalid dry-run flag' >&2; exit 2; }
job_prefix="${PHC2D_JOB_PREFIX:-inv013}"
[[ "$job_prefix" =~ ^[A-Za-z0-9_]+$ ]] || { echo 'invalid job prefix' >&2; exit 2; }
mkdir -p "$PHC2D_RECEIPT_ROOT"
receipt="$PHC2D_RECEIPT_ROOT/submission_${PHC2D_METHOD}.txt"
pending="$PHC2D_RECEIPT_ROOT/submission_${PHC2D_METHOD}.pending"
[[ ! -e "$receipt" ]] || { echo 'receipt exists' >&2; exit 2; }
( set -o noclobber; printf 'manifest=%s\n' "$PHC2D_MANIFEST_SHA256" > "$pending" ) || { echo 'pending submission must be reconciled' >&2; exit 2; }
export PHC2D_ROOT PHC2D_GPU_PLATFORM PHC2D_MANIFEST PHC2D_MANIFEST_SHA256 PHC2D_SOURCE_DIRECTORY
export PHC2D_CHECKPOINT PHC2D_METHOD PHC2D_OUTPUT_ROOT PHC2D_EXPECTED_COMMIT
job_id="$(sbatch --parsable --array=0-2 -p gpu_4090 --gres=gpu:1 -c 6 -t 01:00:00 \
  -J "${job_prefix}_${PHC2D_METHOD}" -o "$PHC2D_RECEIPT_ROOT/${PHC2D_METHOD}_%A_%a.out" \
  -e "$PHC2D_RECEIPT_ROOT/${PHC2D_METHOD}_%A_%a.err" --export=ALL \
  "$worker")"
[[ "$job_id" =~ ^[0-9]+([\;][A-Za-z0-9_-]+)?$ ]] || { echo 'unexpected job ID; reconcile submission' >&2; exit 2; }
printf 'job_id=%s\nmethod=%s\nplatform=%s\ncommit=%s\nmanifest_sha256=%s\n' \
  "$job_id" "$PHC2D_METHOD" "$PHC2D_GPU_PLATFORM" "$PHC2D_EXPECTED_COMMIT" "$PHC2D_MANIFEST_SHA256" > "$receipt"
printf 'worker=%s\nworker_sha256=%s\ncontroller_commit=%s\n' "$worker" "$(sha256sum "$worker" | awk '{print $1}')" "${PHC2D_CONTROLLER_COMMIT:-$PHC2D_EXPECTED_COMMIT}" >> "$receipt"
rm "$pending"
printf '%s\n' "$job_id"
