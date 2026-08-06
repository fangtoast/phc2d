#!/usr/bin/env bash
# Create the versioned phc2d Python environment on N40 or N50 login storage.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

PHC2D_GPU_PLATFORM="${PHC2D_GPU_PLATFORM:-n40}"
mkdir -p "$HOME/run/envs" "$HOME/run/.conda/pkgs"
export CONDA_PKGS_DIRS="$HOME/run/.conda/pkgs"

case "$PHC2D_GPU_PLATFORM" in
  n40)
    module load miniforge/24.1.2
    PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu124}"
    TORCH_VERSION=2.6.0
    TORCH_INDEX=https://download.pytorch.org/whl/cu124
    ;;
  n50)
    module load miniforge/25.3.0-3
    PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu128}"
    TORCH_VERSION=2.7.1
    TORCH_INDEX=https://download.pytorch.org/whl/cu128
    ;;
  *)
    echo "Unsupported PHC2D_GPU_PLATFORM: $PHC2D_GPU_PLATFORM" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PHC2D_ENV/bin/python" ]]; then
  conda create -y -p "$PHC2D_ENV" python=3.10 pip
fi
"$PHC2D_ENV/bin/python" -m pip install --upgrade pip
"$PHC2D_ENV/bin/python" -m pip install --no-cache-dir \
  "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
if [[ "$PHC2D_GPU_PLATFORM" == n50 ]]; then
  "$PHC2D_ENV/bin/python" -m pip install --no-cache-dir \
    nvidia-nccl-cu12==2.27.5 --force-reinstall --no-deps \
    -i https://mirrors.cernet.edu.cn/pypi/web/simple
fi
"$PHC2D_ENV/bin/python" -m pip install --no-cache-dir \
  numpy scipy matplotlib PyYAML h5py tqdm joblib psutil pytest
"$PHC2D_ENV/bin/python" -m pip freeze > "$PHC2D_ENV/requirements.lock.txt"
"$PHC2D_ENV/bin/python" -c "import h5py, torch; print(torch.__version__, torch.version.cuda, h5py.__version__)"
