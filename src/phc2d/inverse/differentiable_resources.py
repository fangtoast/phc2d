"""Prepare the bounded physical allocation for a locked inverse batch.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


def plan_physical_resources(manifest: Mapping[str, Any], resources: Mapping[str, Any], *,
                            manifest_sha256: str) -> dict[str, Any]:
    """Map deduplicated cases to bounded M9 shards without submitting jobs."""
    physical = resources['physical_replay']
    rows = manifest['rows']
    if (not isinstance(rows, list) or len(rows) > physical['max_candidate_roles']
            or not isinstance(manifest.get('selection_lock_sha256'), str)):
        raise ValueError('manifest does not fit the locked origin-batch budget')
    cases_per_shard = physical['cases_per_shard']
    workers = physical['workers_per_shard']
    cpus = physical['cpus_per_shard']
    memory = physical['memory_mib_per_shard']
    if (min(cases_per_shard, workers, cpus, memory, physical['wall_seconds_per_shard']) <= 0
            or workers > cpus or workers > cases_per_shard
            or workers * physical['worker_memory_mib'] + physical['setup_reserve_mib'] > memory):
        raise ValueError('invalid physical worker or resource budget')
    count = math.ceil(len(rows) / cases_per_shard)
    core_hours = count * cpus * physical['wall_seconds_per_shard'] / 3600
    if count > physical['max_shards'] or core_hours > physical['max_allocated_core_hours']:
        raise ValueError('physical allocation exceeds configured budget')
    limit = physical.get('max_concurrent_shards')
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= physical['max_shards']):
        raise ValueError('invalid physical concurrency limit')
    shards = [{'shard': i, 'start': i*cases_per_shard,
               'end': min(len(rows), (i+1)*cases_per_shard)} for i in range(count)]
    result = {'status': 'prepared_not_submitted', 'manifest_sha256': manifest_sha256,
            'selection_lock_sha256': manifest['selection_lock_sha256'],
            'case_count': len(rows), 'shards': shards, 'workers_per_shard': workers,
            'cpus_per_shard': cpus, 'memory_mib_per_shard': memory,
            'worker_memory_mib': physical['worker_memory_mib'],
            'setup_reserve_mib': physical['setup_reserve_mib'],
            'wall_seconds_per_shard': physical['wall_seconds_per_shard'],
            'max_allocated_core_hours': core_hours, 'automatic_retry': False,
            'wsl_crosscheck': resources['wsl_crosscheck']}
    if limit is not None:
        result['max_concurrent_shards'] = min(limit, count)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'resources', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    manifest_raw, resources_raw = Path(args.manifest).read_bytes(), Path(args.resources).read_bytes()
    plan = plan_physical_resources(json.loads(manifest_raw), json.loads(resources_raw),
                                   manifest_sha256=hashlib.sha256(manifest_raw).hexdigest())
    plan['resources_sha256'] = hashlib.sha256(resources_raw).hexdigest()
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
