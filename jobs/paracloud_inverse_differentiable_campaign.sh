#!/usr/bin/env bash
# Run three paired initializations for one source and one inverse method.
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
: "${PHC2D_METHOD:?Set the registered design method}"
: "${PHC2D_OUTPUT_ROOT:?Set trial output directory}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployed code identity}"
: "${SLURM_ARRAY_TASK_ID:?Set source shard index}"
case "$PHC2D_GPU_PLATFORM:$PHC2D_METHOD" in
  n40:adam|n56:de|n56:random|n40:nominal|n40:mean|n40:worst|n56:nominal|n56:mean|n56:worst) ;;
  n56:adam) [[ "${PHC2D_N56_TAKEOVER:-0}" == 1 ]] || { echo 'explicit N56 takeover required' >&2; exit 2; } ;;
  *) echo 'unsupported method/platform assignment' >&2; exit 2 ;;
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
source_id="$(python -m phc2d.inverse.campaign_source --manifest "$PHC2D_MANIFEST" --index "$SLURM_ARRAY_TASK_ID")"
printf -v source_path '%s/source_%04d.json' "$PHC2D_SOURCE_DIRECTORY" "$source_id"
python -m phc2d.inverse.differentiable_campaign_runner --manifest "$PHC2D_MANIFEST" \
  --source "$source_path" --source-id "$source_id" --checkpoint "$PHC2D_CHECKPOINT" \
  --method "$PHC2D_METHOD" --device cuda:0 --output "$PHC2D_OUTPUT_ROOT"
