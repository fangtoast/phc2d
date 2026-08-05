# Shared runtime setup for phc2d jobs on Parallel Computing Cloud M9.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail

: "${SLURM_JOB_ID:?This script must run inside a Slurm allocation.}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required.}"

export PHC2D_CLUSTER_ID="${PHC2D_CLUSTER_ID:-paracloud_bscc_m9}"
PHC2D_ROOT="${PHC2D_ROOT:-/publicfs10/fs10-m9/home/m9s005801/phc2d}"
PHC2D_ENV="${PHC2D_ENV:-/publicfs10/fs10-m9/home/m9s005801/envs/meep_env_paracloud_m9}"

if [[ ! -f "$PHC2D_ENV/bin/activate" ]]; then
  echo "Missing packed environment: $PHC2D_ENV" >&2
  exit 2
fi
if [[ ! -d "$PHC2D_ROOT/src/phc2d" ]]; then
  echo "Missing phc2d checkout: $PHC2D_ROOT" >&2
  exit 2
fi
if [[ -f "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT" ]]; then
  export PHC2D_DEPLOYED_GIT_COMMIT="$(tr -d '[:space:]' < "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT")"
elif git -C "$PHC2D_ROOT" rev-parse HEAD >/dev/null 2>&1; then
  export PHC2D_DEPLOYED_GIT_COMMIT="$(git -C "$PHC2D_ROOT" rev-parse HEAD)"
else
  echo "Missing deployed Git commit provenance." >&2
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
python -c "import meep; from meep import mpb; assert meep.__version__ == '1.33.0-beta'; assert mpb.__version__ == '1.12.0'; print('Meep/MPB:', meep.__version__, mpb.__version__)"
