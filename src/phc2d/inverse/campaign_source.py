"""Resolve a scheduler shard to a frozen campaign source identity.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def source_id_for_shard(manifest: Mapping[str, Any], index: int) -> int:
    """Return the registered source at a zero-based scheduler array index."""
    ids = [row['source_id'] for row in manifest['sources']]
    if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(ids)
            or len(ids) != len(set(ids))
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids)):
        raise ValueError('invalid source shard index or roster')
    return ids[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--index', required=True, type=int)
    args = parser.parse_args()
    print(source_id_for_shard(json.loads(Path(args.manifest).read_bytes()), args.index))


if __name__ == '__main__':
    main()
