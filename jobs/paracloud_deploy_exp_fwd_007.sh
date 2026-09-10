#!/usr/bin/env bash
# Deploy one committed EXP-FWD-007 archive to an isolated N40 staging directory.
# Author: fangtoast
# Email: fangtoast@foxmail.com
# Copyright (c) 2026 fangtoast. All rights reserved.

set -euo pipefail
commit="${1:?usage: $0 FULL_GIT_COMMIT}"; PHC2D_ROOT="${PHC2D_ROOT:-$HOME/run/phc2d}"; PHC2D_ENV="${PHC2D_ENV:-$HOME/run/envs/phc2d_gpu_cu118}"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "a full 40-character commit is required" >&2; exit 2; }
[[ -d "$PHC2D_ROOT/.git" ]] || { echo "deployment must start from a git checkout" >&2; exit 3; }
[[ -x "$PHC2D_ENV/bin/python" ]] || { echo "N40 Python runtime missing" >&2; exit 3; }
actual="$(git -C "$PHC2D_ROOT" rev-parse "$commit")"; [[ "$actual" == "$commit" ]] || { echo "commit is not present locally" >&2; exit 3; }
stage="$(mktemp -d "${TMPDIR:-/tmp}/phc2d-exp-fwd-007.XXXXXX")"; archive="$stage/phc2d.tar"
cleanup() { rm -rf "$stage"; }; trap cleanup EXIT
git -C "$PHC2D_ROOT" archive --format=tar "$commit" > "$archive"
archive_sha="$(sha256sum "$archive" | awk '{print $1}')"; tar -xf "$archive" -C "$stage"
[[ "$(git -C "$stage" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]] || true
PYTHONPATH="$stage/src" "$PHC2D_ENV/bin/python" -m compileall -q "$stage/src/phc2d"
for script in "$stage"/jobs/paracloud_*exp_fwd_007*.sh "$stage"/jobs/paracloud_full_cell_k5_train.sh "$stage"/jobs/paracloud_full_cell_six_domain*.sh; do bash -n "$script"; done
# rsync intentionally has no --delete: data, runs, checkpoints, and historical misdelivery directories survive.
rsync -a --exclude='.git' --exclude='data/' --exclude='runs/' --exclude='checkpoints/' "$stage/" "$PHC2D_ROOT/"
printf '%s\n' "$commit" > "$PHC2D_ROOT/DEPLOYED_GIT_COMMIT"
printf '%s\n' "$commit" > "$PHC2D_ROOT/.deployed_git_commit"
printf 'commit=%s\narchive_sha256=%s\ndeployed_at=%s\n' "$commit" "$archive_sha" "$(date -Is)" > "$PHC2D_ROOT/deployment_manifest_EXP-FWD-007.env"
