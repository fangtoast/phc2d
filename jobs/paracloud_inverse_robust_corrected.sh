#!/usr/bin/env bash
# Run one source/mode using frozen manufacturing feedback on a GPU.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment root}"
: "${PHC2D_GPU_PLATFORM:?Set n40 or n56}"
: "${PHC2D_MANIFEST:?Set frozen robust manifest}"
: "${PHC2D_MANIFEST_SHA256:?Set robust manifest hash}"
: "${PHC2D_SOURCE_DIRECTORY:?Set frozen source directory}"
: "${PHC2D_FEEDBACK:?Set physical feedback observations}"
: "${PHC2D_FEEDBACK_SHA256:?Set feedback hash}"
: "${PHC2D_CHECKPOINT:?Set frozen checkpoint}"
: "${PHC2D_OUTPUT_ROOT:?Set output directory}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployed code identity}"
: "${PHC2D_METHOD:?Set corrected mode}"
: "${PHC2D_RADIUS:?Set registered local radius}"
: "${PHC2D_RIDGE:?Set registered residual ridge}"
: "${SLURM_ARRAY_TASK_ID:?Set source shard index}"
case "$PHC2D_METHOD" in
  nominal_corrected) mode=nominal ;;
  mean_corrected) mode=mean ;;
  worst_corrected) mode=worst ;;
  *) echo 'invalid corrected robust method' >&2; exit 2 ;;
esac
if [[ "$PHC2D_GPU_PLATFORM" == n56 ]]; then
  export PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1}"
fi
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  set +u
  source /etc/profile.d/modules.sh
  set -u
fi
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'deployment mismatch' >&2; exit 2; }
printf '%s  %s\n' "$PHC2D_MANIFEST_SHA256" "$PHC2D_MANIFEST" | sha256sum -c -
printf '%s  %s\n' "$PHC2D_FEEDBACK_SHA256" "$PHC2D_FEEDBACK" | sha256sum -c -
source_id="$(python -m phc2d.inverse.campaign_source --manifest "$PHC2D_MANIFEST" --index "$SLURM_ARRAY_TASK_ID")"
printf -v source_path '%s/source_%04d.json' "$PHC2D_SOURCE_DIRECTORY" "$source_id"
python -m phc2d.inverse.differentiable_robust_corrected_runner --manifest "$PHC2D_MANIFEST" \
  --feedback "$PHC2D_FEEDBACK" --source "$source_path" --source-id "$source_id" \
  --checkpoint "$PHC2D_CHECKPOINT" --mode "$mode" --radius "$PHC2D_RADIUS" --ridge "$PHC2D_RIDGE" \
  --device cuda:0 --output "$PHC2D_OUTPUT_ROOT"
