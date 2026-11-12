#!/bin/bash -l
# Isolated EXP-FWD-031 frozen-checkpoint test worker.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.
set -euo pipefail
: "${PHC2D_SUITE:?}"
: "${PHC2D_CASE:?}"
: "${PHC2D_TARGET:?}"
: "${PHC2D_DATASET:?}"
: "${PHC2D_DATASET_SHA256:?}"
: "${PHC2D_GRID_FIELD:?}"
: "${PHC2D_TEST_SOURCE:?}"
: "${PHC2D_DEPLOY_ROOT:?}"
: "${PHC2D_ARCHIVE_SHA256:?}"
: "${PHC2D_GPU_PLATFORM:?}"
printf '%s  %s\n' "$PHC2D_ARCHIVE_SHA256" "$PHC2D_DEPLOY_ROOT/deployment.tar.gz" | sha256sum -c -
task_root=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/phc2d-031-test-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$task_root"' EXIT
tar -xzf "$PHC2D_DEPLOY_ROOT/deployment.tar.gz" -C "$task_root"
cp "$PHC2D_DEPLOY_ROOT/DEPLOYED_GIT_COMMIT" "$task_root/DEPLOYED_GIT_COMMIT"
cp "$PHC2D_TEST_SOURCE" "$task_root/src/phc2d/forward/official_comparison_test.py"
sha256sum "$task_root/src/phc2d/forward/official_comparison_test.py"
case "$PHC2D_GPU_PLATFORM" in
  n40)
    export PHC2D_ENV=/HOME/scw6g7a/run/envs/phc2d_gpu_cu118
    official_overlay=/data/run01/scw6g7a/envs/phc2d_exp_fwd_031/official_overlay ;;
  n56)
    export PHC2D_ENV=/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1
    official_overlay=/data/run01/scxl892/envs/phc2d_exp_fwd_031/official_overlay ;;
  *) exit 2 ;;
esac
tr -d '\r' < "$task_root/jobs/official_neuraloperator_2_0_0.sha256" > "$task_root/official_source.sha256"
(cd "$official_overlay" && sha256sum -c "$task_root/official_source.sha256")
export PHC2D_ROOT="$task_root"
source "$task_root/jobs/paracloud_gpu_common.sh"
export PYTHONPATH="$official_overlay:$PYTHONPATH"
python -c 'import importlib.metadata as m; assert m.version("neuraloperator")=="2.0.0"; assert m.version("tensorly")=="0.9.0"'
claim="$PHC2D_SUITE/train_claims/$(basename "$PHC2D_SUITE")_${PHC2D_CASE}/completed.json"
args=(--run "$PHC2D_SUITE/training/$PHC2D_CASE" --completed "$claim" --dataset "$PHC2D_DATASET" --dataset-sha256 "$PHC2D_DATASET_SHA256" --grid-field "$PHC2D_GRID_FIELD" --output "$PHC2D_SUITE/test/$PHC2D_CASE/$PHC2D_TARGET")
if [[ "$PHC2D_TARGET" == dat* ]]; then args+=(--external); fi
python -u -m phc2d.forward.official_comparison_test "${args[@]}"
