#!/usr/bin/env bash
# Deploy a frozen EXP-FWD-020 commit into an immutable N56 version directory.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
commit="${1:?usage: $0 FULL_GIT_COMMIT}"
source_root="${PHC2D_SOURCE_ROOT:-/data/run01/scxl892/phc2d}"
deployment_root="${PHC2D_DEPLOYMENT_ROOT:-/data/run01/scxl892/deployments/EXP-FWD-020}"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "full commit required" >&2; exit 2; }
target="$deployment_root/$commit"
[[ ! -e "$target" ]] || { echo "immutable deployment already exists" >&2; exit 3; }
git -C "$source_root" rev-parse --verify "$commit^{commit}" >/dev/null
mkdir -p "$(dirname "$target")"; git -C "$source_root" archive "$commit" | tar -x -C "$(dirname "$target")" --transform="s,^,$commit/,"
printf '%s\n' "$commit" > "$target/DEPLOYED_GIT_COMMIT"
printf 'commit=%s\narchive_sha256=%s\n' "$commit" "$(git -C "$source_root" archive "$commit" | sha256sum | cut -d' ' -f1)" > "$target/deployment_manifest_EXP-FWD-020.env"
chmod -R a-w "$target"; echo "$target"
