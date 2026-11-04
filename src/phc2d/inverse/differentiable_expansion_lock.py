"""Freeze the four-center extension separately from the origin sub-batch.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phc2d.inverse.differentiable_expansion import combine_centers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension-config', required=True)
    parser.add_argument('--shard', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    extension_raw = Path(args.extension_config).read_bytes()
    extension_hash = hashlib.sha256(extension_raw).hexdigest()
    raw_shards = [Path(path).read_bytes() for path in args.shard]
    shards = [json.loads(raw) for raw in raw_shards]
    if any(shard.get('extension_config_sha256') != extension_hash for shard in shards):
        raise ValueError('shards do not match the extension configuration')
    result = combine_centers(shards, json.loads(extension_raw))
    result['shard_sha256_by_center'] = {
        str(shard['center_index']): hashlib.sha256(raw).hexdigest() for shard, raw in zip(shards, raw_shards)}
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
