#!/usr/bin/env bash
# Perform the mandatory single-5090 EXP-FWD-020 preflight before G1 submission.
# Author: fangtoast
# Email: fangtoast
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d-exp-fwd-020}"
export PHC2D_GPU_PLATFORM=n56
export PHC2D_CUDA_MODULE=cuda/12.8
export PHC2D_ENV="${PHC2D_ENV:-/data/run01/scxl892/envs/phc2d_gpu_cu128_torch271_r1}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_PREFLIGHT_CONFIG:?Set a frozen G1-derived runner config.}"
: "${PHC2D_PREFLIGHT_RUN_DIR:?Set standard preflight run directory.}"
python -c 'import h5py, torch; assert torch.cuda.is_available(); assert torch.__version__.startswith("2.7.1"), torch.__version__; assert torch.version.cuda and torch.version.cuda.startswith("12.8"), torch.version.cuda; print(torch.__version__, torch.version.cuda, h5py.__version__)'
python -m phc2d.forward.full_cell --config "$PHC2D_PREFLIGHT_CONFIG" --model global_local_band_operator_v1 --seed 0 --admission-probe --run-dir "$PHC2D_PREFLIGHT_RUN_DIR"
for artifact in run.log run.meta.json config.resolved.yaml metrics.json results.json history.csv model.pt model_summary.json; do test -s "$PHC2D_PREFLIGHT_RUN_DIR/$artifact"; done
