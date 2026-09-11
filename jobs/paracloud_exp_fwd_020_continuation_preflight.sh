#!/usr/bin/env bash
# Exercise the frozen epoch-201 warm-start admission for EXP-FWD-020.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d-exp-fwd-020}"
export PHC2D_GPU_PLATFORM=n56
export PHC2D_CUDA_MODULE=cuda/12.8
export PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD020_MANIFEST:?Set frozen continuation manifest.}"
: "${PHC2D_FWD020_TASK_INDEX:?Set frozen representative task index.}"
: "${PHC2D_FWD020_PROBE_RUN:?Set isolated epoch-201 admission output directory.}"
python -c 'import h5py, torch; assert torch.cuda.is_available(); assert torch.__version__.startswith("2.7.1"), torch.__version__; assert torch.version.cuda and torch.version.cuda.startswith("12.8"), torch.version.cuda; print(torch.__version__, torch.version.cuda, h5py.__version__)'
python -m phc2d.forward.global_local_bridge_campaign run-training-task \
  --manifest "$PHC2D_FWD020_MANIFEST" --task-index "$PHC2D_FWD020_TASK_INDEX" \
  --repository-root "$PHC2D_ROOT" --admission-probe --run-dir-override "$PHC2D_FWD020_PROBE_RUN"
for artifact in run.log run.meta.json config.resolved.yaml history.csv metrics.json results.json model.pt model_summary.json; do
  test -s "$PHC2D_FWD020_PROBE_RUN/$artifact"
done
grep -q '"stop_epoch": 201' "$PHC2D_FWD020_PROBE_RUN/results.json"
