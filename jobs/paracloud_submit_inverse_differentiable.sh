#!/usr/bin/env bash
# Submit one isolated MVP preflight or registered design shard with a receipt.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment root}"
: "${PHC2D_GPU_PLATFORM:?Set n40 or n56}"
: "${PHC2D_ROLE:?Set preflight, directions, adam, de or random}"
: "${PHC2D_SOURCE:?Set source path}"
: "${PHC2D_CHECKPOINT:?Set checkpoint path}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployed commit}"
: "${PHC2D_BATCH_ROOT:?Set unique batch directory}"
case "$PHC2D_GPU_PLATFORM:$PHC2D_ROLE" in
  n40:preflight|n56:preflight|n40:adam|n40:directions|n56:de|n56:random) ;;
  n56:adam|n56:directions) [[ "${PHC2D_N56_TAKEOVER:-0}" == 1 ]] || { echo 'explicit N56 takeover required' >&2; exit 2; } ;;
  *) echo 'role is not assigned to this platform' >&2; exit 2 ;;
esac
export PHC2D_CONFIG="${PHC2D_CONFIG:-$PHC2D_ROOT/configs/research/exp_inv_012_origin_v1.json}"
export PHC2D_CONFIG_SHA256="$(sha256sum "$PHC2D_CONFIG" | awk '{print $1}')"
export PHC2D_OUTPUT="$PHC2D_BATCH_ROOT/$PHC2D_ROLE.json"
[[ -f "$PHC2D_SOURCE" && -f "$PHC2D_CHECKPOINT" ]] || { echo 'missing source or checkpoint' >&2; exit 2; }
[[ ! -e "$PHC2D_OUTPUT" ]] || { echo 'output already exists' >&2; exit 2; }
[[ "$(tr -d '[:space:]' < "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT")" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'deployment mismatch' >&2; exit 2; }
if [[ "$PHC2D_ROLE" != preflight ]]; then
  : "${PHC2D_PREFLIGHT_GATE:?Set preflight gate path}"
  : "${PHC2D_PREFLIGHT_GATE_SHA256:?Set preflight gate SHA-256}"
  [[ -f "$PHC2D_PREFLIGHT_GATE" ]] || { echo 'preflight gate missing' >&2; exit 2; }
  printf '%s  %s\n' "$PHC2D_PREFLIGHT_GATE_SHA256" "$PHC2D_PREFLIGHT_GATE" | sha256sum -c - >/dev/null
  python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["status"] == "passed" and x["config_sha256"] == sys.argv[2] and x["code_commit"] == sys.argv[3]' \
    "$PHC2D_PREFLIGHT_GATE" "$PHC2D_CONFIG_SHA256" "$PHC2D_EXPECTED_COMMIT"
fi
case "${PHC2D_SUBMIT_DRY_RUN:-0}" in
  1)
    printf 'prepared platform=%s role=%s partition=gpu_4090 gpus=1 cpus=6 wall_minutes=30 config_sha256=%s\n' \
      "$PHC2D_GPU_PLATFORM" "$PHC2D_ROLE" "$PHC2D_CONFIG_SHA256"
    exit 0 ;;
  0) ;;
  *) echo 'invalid dry-run flag' >&2; exit 2 ;;
esac
mkdir -p "$PHC2D_BATCH_ROOT"
receipt="$PHC2D_BATCH_ROOT/submission_${PHC2D_ROLE}.txt"
pending="$PHC2D_BATCH_ROOT/submission_${PHC2D_ROLE}.pending"
[[ ! -e "$receipt" ]] || { echo 'submission receipt already exists' >&2; exit 2; }
( set -o noclobber; printf 'role=%s\n' "$PHC2D_ROLE" > "$pending" ) || { echo 'submission pending; reconcile before retry' >&2; exit 2; }
export PHC2D_ROOT PHC2D_GPU_PLATFORM PHC2D_ROLE PHC2D_SOURCE PHC2D_CHECKPOINT PHC2D_EXPECTED_COMMIT
job_id="$(sbatch --parsable -p gpu_4090 --gres=gpu:1 -c 6 -t 00:30:00 \
  -J "inv012_${PHC2D_ROLE}" -o "$PHC2D_BATCH_ROOT/${PHC2D_ROLE}_%j.out" \
  -e "$PHC2D_BATCH_ROOT/${PHC2D_ROLE}_%j.err" --export=ALL \
  "$PHC2D_ROOT/jobs/paracloud_inverse_differentiable_mvp.sh")"
[[ "$job_id" =~ ^[0-9]+([\;][A-Za-z0-9_-]+)?$ ]] || { echo 'unexpected submission result; reconcile before retry' >&2; exit 2; }
printf 'job_id=%s\nrole=%s\nplatform=%s\ncommit=%s\nconfig_sha256=%s\n' \
  "$job_id" "$PHC2D_ROLE" "$PHC2D_GPU_PLATFORM" "$PHC2D_EXPECTED_COMMIT" "$PHC2D_CONFIG_SHA256" > "$receipt"
rm "$pending"
printf '%s\n' "$job_id"
