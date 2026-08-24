#!/usr/bin/env bash
# Run the EXP-FWD-007 six-domain GPU identity preflight on one N40 GPU.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
: "${PHC2D_FWD007_SIX_DOMAIN_CAMPAIGN:?campaign path required}"
PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_six_domain_n40 preflight \
  --campaign-dir "$PHC2D_FWD007_SIX_DOMAIN_CAMPAIGN" \
  --repository-root "$PHC2D_ROOT" \
  --device cuda:0
