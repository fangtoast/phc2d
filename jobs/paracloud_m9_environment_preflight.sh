#!/usr/bin/env bash
# Unpack and verify the offline phc2d environment on a Parallel Computing Cloud M9 node.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paracloud_m9_common.sh"

echo "host=$(hostname)"
cat /etc/os-release
ldd --version | head -n 1
python -c "import platform; import meep; from meep import mpb; print(platform.libc_ver()); print(meep.__version__, mpb.__version__)"
