# Shared runtime setup for phc2d jobs on the Xingyi cluster.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

: "${SLURM_JOB_ID:?This script must run inside a Slurm allocation.}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required.}"

PHC2D_ROOT="${PHC2D_ROOT:-$HOME/HDD_POOL/phc2d}"
PHC2D_ENV="${PHC2D_ENV:-$HOME/HDD_POOL/envs/meep_env_xingyi}"

if [[ ! -f "$PHC2D_ENV/bin/activate" ]]; then
  echo "Missing packed environment: $PHC2D_ENV" >&2
  exit 2
fi
if [[ ! -d "$PHC2D_ROOT/src/phc2d" ]]; then
  echo "Missing phc2d checkout: $PHC2D_ROOT" >&2
  exit 2
fi

module purge >/dev/null 2>&1 || true
source "$PHC2D_ENV/bin/activate"
export PYTHONPATH="$PHC2D_ROOT/src"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$PHC2D_ROOT"
python -c "import meep; from meep import mpb; mpb_version = '.'.join(map(str, mpb.__version__)) if isinstance(mpb.__version__, (tuple, list)) else str(mpb.__version__); assert meep.__version__ == '1.33.0-beta'; assert mpb_version == '1.12.0'; print('Meep/MPB:', meep.__version__, mpb_version)"
