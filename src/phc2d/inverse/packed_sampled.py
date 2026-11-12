"""Run sampled MPB screening cases in one bounded allocation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phc2d.inverse.packed_cases import run_packed_cases


def main() -> None:
    parser = argparse.ArgumentParser(description='Run bounded sampled inverse screening cases')
    for name in ('manifest', 'manifest-sha256', 'output-root'):
        parser.add_argument('--' + name, required=True)
    for name in ('start', 'end', 'workers', 'cpu-limit', 'memory-budget-mib', 'worker-memory-mib', 'setup-reserve-mib'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--worker-resource-report', action='store_true')
    args = parser.parse_args()
    result = run_packed_cases(mode='sampled', manifest=Path(args.manifest),
        manifest_sha256=args.manifest_sha256, output_root=Path(args.output_root),
        start=args.start, end=args.end, workers=args.workers, cpu_limit=args.cpu_limit,
        memory_budget_mib=args.memory_budget_mib, worker_memory_mib=args.worker_memory_mib,
        setup_reserve_mib=args.setup_reserve_mib, worker_resource_report=args.worker_resource_report)
    print(json.dumps(result))
    if not result['all_succeeded']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
