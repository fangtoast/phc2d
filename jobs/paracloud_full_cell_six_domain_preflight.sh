#!/usr/bin/env bash
# Run a frozen six-domain GPU identity preflight on one N40 GPU.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"
export PHC2D_DEFER_CUDA_CHECK_TO_RUNNER=1
source "$PHC2D_ROOT/jobs/paracloud_gpu_common.sh"
campaign="${PHC2D_SIX_DOMAIN_CAMPAIGN:-${PHC2D_FWD008_SIX_DOMAIN_CAMPAIGN:-${PHC2D_FWD007_SIX_DOMAIN_CAMPAIGN:-}}}"
: "${campaign:?campaign path required}"
PYTHONPATH="$PHC2D_ROOT/src" python -m phc2d.forward.full_cell_six_domain_n40 preflight \
  --campaign-dir "$campaign" \
  --repository-root "$PHC2D_ROOT" \
  --device cuda:0
