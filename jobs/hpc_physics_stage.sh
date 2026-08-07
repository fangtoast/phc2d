#!/usr/bin/env bash
# Run one stage of a portable phc2d Slurm physics campaign.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

: "${PHC2D_PLATFORM_ID:?Set PHC2D_PLATFORM_ID.}"
: "${PHC2D_STAGE:?Set PHC2D_STAGE to plan, solve-shard, or finalize.}"
: "${PHC2D_ADAPTER:?Set PHC2D_ADAPTER.}"
: "${PHC2D_CONFIG:?Set PHC2D_CONFIG.}"
: "${PHC2D_RUN_DIR:?Set PHC2D_RUN_DIR.}"
: "${PHC2D_SHARD_COUNT:?Set PHC2D_SHARD_COUNT.}"

case "$PHC2D_PLATFORM_ID" in
  xingyi)
    PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
    source "$PHC2D_ROOT/jobs/xingyi_common.sh"
    ;;
  paracloud_bscc_m9)
    PHC2D_ROOT="${PHC2D_ROOT:-/publicfs10/fs10-m9/home/m9s005801/phc2d}"
    source "$PHC2D_ROOT/jobs/paracloud_m9_common.sh"
    ;;
  *)
    echo "Unsupported PHC2D_PLATFORM_ID: $PHC2D_PLATFORM_ID" >&2
    exit 2
    ;;
esac

workers="${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required.}"
case "$PHC2D_STAGE" in
  plan)
    python -m phc2d.cli hpc plan \
      --adapter "$PHC2D_ADAPTER" \
      --config "$PHC2D_CONFIG" \
      --run-dir "$PHC2D_RUN_DIR" \
      --workers "$workers" \
      --shards "$PHC2D_SHARD_COUNT" \
      --platform "$PHC2D_PLATFORM_ID"
    ;;
  solve-shard)
    if [[ "${PHC2D_AUTHORIZE_FORMAL:-NO}" != "YES" ]]; then
      echo "Formal shard solving requires PHC2D_AUTHORIZE_FORMAL=YES." >&2
      exit 3
    fi
    : "${SLURM_ARRAY_TASK_ID:?solve-shard requires a Slurm array task.}"
    resume_args=()
    if [[ "${PHC2D_ALLOW_RESUME:-0}" == "1" ]]; then
      resume_args+=(--allow-resume)
    fi
    python -m phc2d.cli hpc solve-shard \
      --adapter "$PHC2D_ADAPTER" \
      --config "$PHC2D_CONFIG" \
      --run-dir "$PHC2D_RUN_DIR" \
      --shard-index "$SLURM_ARRAY_TASK_ID" \
      --shard-count "$PHC2D_SHARD_COUNT" \
      --workers "$workers" \
      "${resume_args[@]}"
    ;;
  finalize)
    if [[ "${PHC2D_AUTHORIZE_FORMAL:-NO}" != "YES" ]]; then
      echo "Formal finalization requires PHC2D_AUTHORIZE_FORMAL=YES." >&2
      exit 3
    fi
    python -m phc2d.cli hpc finalize \
      --adapter "$PHC2D_ADAPTER" \
      --config "$PHC2D_CONFIG" \
      --run-dir "$PHC2D_RUN_DIR" \
      --workers "$workers"
    ;;
  *)
    echo "PHC2D_STAGE must be plan, solve-shard, or finalize." >&2
    exit 2
    ;;
esac
