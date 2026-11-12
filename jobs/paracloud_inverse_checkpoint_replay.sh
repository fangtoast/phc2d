#!/usr/bin/env bash
# Validation-only replay of a frozen single-lane checkpoint.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_ROOT:?Set deployment root}"
: "${PHC2D_EXPECTED_COMMIT:?Set deployment commit}"
: "${PHC2D_DATASET:?Set dataset path}"
: "${PHC2D_DATASET_SHA256:?Set dataset digest}"
: "${PHC2D_GRID_FIELD:?Set checkpoint input field}"
: "${PHC2D_CHECKPOINT:?Set checkpoint path}"
: "${PHC2D_CHECKPOINT_SHA256:?Set checkpoint digest}"
: "${PHC2D_EXPECTED_MODEL:?Set model identity}"
: "${PHC2D_OUTPUT:?Set unique output path}"
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  set +u
  source /etc/profile.d/modules.sh
  set -u
fi
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
[[ "$PHC2D_DEPLOYED_GIT_COMMIT" == "$PHC2D_EXPECTED_COMMIT" ]] || { echo 'deployment mismatch' >&2; exit 2; }
if [[ -n "${PHC2D_OFFICIAL_OVERLAY:-}" ]]; then
  [[ -d "$PHC2D_OFFICIAL_OVERLAY/neuralop" ]] || { echo 'official overlay missing' >&2; exit 2; }
  sed 's/\r$//' "$PHC2D_ROOT/jobs/official_neuraloperator_2_0_0.sha256" | \
    (cd "$PHC2D_OFFICIAL_OVERLAY" && sha256sum -c -)
  export PYTHONPATH="$PHC2D_OFFICIAL_OVERLAY:$PYTHONPATH"
  python -c 'import importlib.metadata as m; assert m.version("neuraloperator") == "2.0.0"; assert m.version("tensorly") == "0.9.0"'
fi
python -m phc2d.inverse.checkpoint_validation_replay --dataset "$PHC2D_DATASET" \
  --dataset-sha256 "$PHC2D_DATASET_SHA256" --grid-field "$PHC2D_GRID_FIELD" \
  --checkpoint "$PHC2D_CHECKPOINT" --checkpoint-sha256 "$PHC2D_CHECKPOINT_SHA256" \
  --expected-model "$PHC2D_EXPECTED_MODEL" --tolerance-pct-points 0.001 \
  --batch-size 16 --device cuda:0 --output "$PHC2D_OUTPUT"
