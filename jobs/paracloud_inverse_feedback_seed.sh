#!/usr/bin/env bash
# Evaluate one source's frozen feedback stencils on a GPU.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment root}"
: "${PHC2D_GPU_PLATFORM:?Set n40 or n56}"
: "${PHC2D_MANIFEST:?Set pure-design manifest}"
: "${PHC2D_MANIFEST_SHA256:?Set design manifest hash}"
: "${PHC2D_DESIGN_LOCK:?Set pure-design selection lock}"
: "${PHC2D_DESIGN_LOCK_SHA256:?Set design lock hash}"
: "${PHC2D_BASE_CONFIG:?Set original model configuration}"
: "${PHC2D_FEEDBACK_CONFIG:?Set feedback campaign configuration}"
: "${PHC2D_FEEDBACK_CONFIG_SHA256:?Set feedback configuration hash}"
: "${PHC2D_SOURCE_DIRECTORY:?Set source directory}"
: "${PHC2D_CHECKPOINT:?Set frozen checkpoint}"
: "${PHC2D_OUTPUT_ROOT:?Set output directory}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployed code identity}"
: "${SLURM_ARRAY_TASK_ID:?Set source shard index}"
case "$SLURM_ARRAY_TASK_ID" in
  0) source_id=8 ;;
  1) source_id=491 ;;
  2) source_id=348 ;;
  *) echo 'invalid source shard index' >&2; exit 2 ;;
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
printf '%s  %s\n' "$PHC2D_DESIGN_LOCK_SHA256" "$PHC2D_DESIGN_LOCK" | sha256sum -c -
printf '%s  %s\n' "$PHC2D_FEEDBACK_CONFIG_SHA256" "$PHC2D_FEEDBACK_CONFIG" | sha256sum -c -
printf -v source_path '%s/source_%04d.json' "$PHC2D_SOURCE_DIRECTORY" "$source_id"
python -m phc2d.inverse.differentiable_feedback_seed --campaign "$PHC2D_FEEDBACK_CONFIG" \
  --base "$PHC2D_BASE_CONFIG" --design-manifest "$PHC2D_MANIFEST" --design-lock "$PHC2D_DESIGN_LOCK" \
  --source "$source_path" --source-id "$source_id" --checkpoint "$PHC2D_CHECKPOINT" \
  --device cuda:0 --output "$PHC2D_OUTPUT_ROOT"
