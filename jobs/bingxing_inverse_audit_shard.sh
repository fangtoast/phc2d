#!/usr/bin/env bash
# Run a contiguous frozen inverse audit shard inside one M9 allocation.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment root}"
: "${PHC2D_MANIFEST:?Set immutable candidate manifest}"
: "${PHC2D_MANIFEST_SHA256:?Set candidate manifest SHA-256}"
: "${PHC2D_OUTPUT_ROOT:?Set unique output directory}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployed source commit}"
: "${PHC2D_CASE_COUNT:?Set case count}"
: "${PHC2D_CASES_PER_SHARD:?Set cases per shard}"
: "${SLURM_ARRAY_TASK_ID:?Set shard index}"
source "$PHC2D_ROOT/jobs/paracloud_m9_common.sh"
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'deployment mismatch' >&2; exit 2; }
start=$((SLURM_ARRAY_TASK_ID * PHC2D_CASES_PER_SHARD))
end=$((start + PHC2D_CASES_PER_SHARD))
if (( end > PHC2D_CASE_COUNT )); then end=$PHC2D_CASE_COUNT; fi
workers="${PHC2D_WORKERS_PER_SHARD:-1}"
if (( workers > 1 )); then
  : "${SLURM_CPUS_PER_TASK:?CPU allocation required}"
  : "${SLURM_MEM_PER_NODE:?node memory allocation required}"
  : "${PHC2D_WORKER_MEMORY_MIB:?measured per-worker memory budget required}"
  : "${PHC2D_SETUP_RESERVE_MIB:?measured shared memory reserve required}"
  resource_flags=()
  case "${PHC2D_RECORD_WORKER_RESOURCES:-0}" in
    0) ;;
    1) resource_flags+=(--worker-resource-report) ;;
    *) echo 'PHC2D_RECORD_WORKER_RESOURCES must be 0 or 1' >&2; exit 2 ;;
  esac
  python -m phc2d.inverse.packed_sampled --manifest "$PHC2D_MANIFEST" \
    --manifest-sha256 "$PHC2D_MANIFEST_SHA256" --output-root "$PHC2D_OUTPUT_ROOT" \
    --start "$start" --end "$end" --workers "$workers" --cpu-limit "$SLURM_CPUS_PER_TASK" \
    --memory-budget-mib "$SLURM_MEM_PER_NODE" --worker-memory-mib "$PHC2D_WORKER_MEMORY_MIB" \
    --setup-reserve-mib "$PHC2D_SETUP_RESERVE_MIB" "${resource_flags[@]}"
  exit $?
fi
for ((index=start; index<end && index<PHC2D_CASE_COUNT; index++)); do
  python -m phc2d.inverse.audit_mpb --manifest "$PHC2D_MANIFEST" \
    --manifest-sha256 "$PHC2D_MANIFEST_SHA256" --case-index "$index" \
    --output-root "$PHC2D_OUTPUT_ROOT"
done
