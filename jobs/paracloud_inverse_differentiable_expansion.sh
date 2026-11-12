#!/usr/bin/env bash
# Run one preregistered neighbor-center direction diagnostic on N40/N56.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set isolated deployment root}"
: "${PHC2D_CONFIG:?Set base configuration}"
: "${PHC2D_CONFIG_SHA256:?Set base configuration SHA-256}"
: "${PHC2D_EXTENSION:?Set extension configuration}"
: "${PHC2D_EXTENSION_SHA256:?Set extension configuration SHA-256}"
: "${PHC2D_SOURCE:?Set source path}"
: "${PHC2D_CHECKPOINT:?Set checkpoint path}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployed commit}"
: "${PHC2D_PREFLIGHT_GATE:?Set preflight gate}"
: "${PHC2D_PREFLIGHT_GATE_SHA256:?Set preflight gate SHA-256}"
: "${PHC2D_OUTPUT_ROOT:?Set unique output directory}"
center="${PHC2D_EXTENSION_CENTER_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
[[ "$center" =~ ^[0-3]$ ]] || { echo 'center index must be 0 through 3' >&2; exit 2; }
if [[ "${PHC2D_GPU_PLATFORM:-}" == n56 ]]; then
  export PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1}"
fi
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  set +u
  source /etc/profile.d/modules.sh
  set -u
fi
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'deployment mismatch' >&2; exit 2; }
printf '%s  %s\n' "$PHC2D_CONFIG_SHA256" "$PHC2D_CONFIG" | sha256sum -c -
printf '%s  %s\n' "$PHC2D_EXTENSION_SHA256" "$PHC2D_EXTENSION" | sha256sum -c -
printf '%s  %s\n' "$PHC2D_PREFLIGHT_GATE_SHA256" "$PHC2D_PREFLIGHT_GATE" | sha256sum -c -
python -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["status"] == "passed" and x["config_sha256"] == sys.argv[2] and x["code_commit"] == sys.argv[3]' \
  "$PHC2D_PREFLIGHT_GATE" "$PHC2D_CONFIG_SHA256" "$PHC2D_EXPECTED_COMMIT"
python -m phc2d.inverse.differentiable_expansion --base-config "$PHC2D_CONFIG" \
  --extension-config "$PHC2D_EXTENSION" --source "$PHC2D_SOURCE" --checkpoint "$PHC2D_CHECKPOINT" \
  --center-index "$center" --device cuda:0 --output "$PHC2D_OUTPUT_ROOT/center_${center}.json"
