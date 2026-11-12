#!/bin/bash -l
# Evaluate existing predictions or replay frozen checkpoints into new sidecars.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside an authorized scheduler allocation}"
: "${PHC2D_ROOT:?Immutable deployed source snapshot}"
: "${PHC2D_ENV:?Python environment}"
: "${PHC2D_METRIC_MANIFEST:?Frozen JSON manifest}"
: "${PHC2D_METRIC_OUTPUT:?New output directory}"
: "${PHC2D_METRIC_MODE:?existing or replay}"
export PYTHONPATH="${PHC2D_OFFICIAL_OVERLAY:+$PHC2D_OFFICIAL_OVERLAY:}$PHC2D_ROOT/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
case "$PHC2D_METRIC_MODE" in
  existing) module=phc2d.forward.spectral_metric_evaluation ;;
  replay)
    : "${PHC2D_OFFICIAL_OVERLAY:?Frozen NeuralOperator overlay required}"
    source /etc/profile.d/modules.sh
    source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
    export PYTHONPATH="$PHC2D_OFFICIAL_OVERLAY:$PHC2D_ROOT/src"
    tr -d '\r' < "$PHC2D_ROOT/jobs/official_neuraloperator_2_0_0.sha256" | (cd "$PHC2D_OFFICIAL_OVERLAY" && sha256sum -c -)
    "$PHC2D_ENV/bin/python" -c 'import importlib.metadata as m; assert m.version("neuraloperator")=="2.0.0"; assert m.version("tensorly")=="0.9.0"'
    module=phc2d.forward.frozen_spectral_replay ;;
  *) echo "Unknown metric mode" >&2; exit 2 ;;
esac
"$PHC2D_ENV/bin/python" -u -m "$module" --manifest "$PHC2D_METRIC_MANIFEST" --output "$PHC2D_METRIC_OUTPUT"
