"""Run independent inverse refinement cases in a bounded allocation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_packed_refinement(*, manifest: Path, manifest_sha256: str, protocol: Path,
                          protocol_sha256: str, output_root: Path, start: int, end: int,
                          workers: int, cpu_limit: int, memory_budget_mib: int,
                          worker_memory_mib: int, setup_reserve_mib: int,
                          worker_resource_report: bool = False) -> dict:
    """Run disjoint cases; caller establishes memory estimates and scheduler limits.

    Each worker is a single-threaded subprocess. Failure is recorded per case;
    existing output is never reused or overwritten. No solver imports occur here.
    """
    if not isinstance(worker_resource_report, bool):
        raise ValueError('worker resource reporting must be boolean')
    if worker_resource_report and not os.access('/usr/bin/time', os.X_OK):
        raise ValueError('worker resource reporting requires GNU /usr/bin/time')
    numbers = (start, end, workers, cpu_limit, memory_budget_mib, worker_memory_mib, setup_reserve_mib)
    if any(isinstance(n, bool) or not isinstance(n, int) for n in numbers):
        raise ValueError('integer resource and case bounds required')
    if (start < 0 or end <= start or workers < 1 or workers > cpu_limit
            or memory_budget_mib < 1 or worker_memory_mib < 1 or setup_reserve_mib < 0
            or workers * worker_memory_mib + setup_reserve_mib > memory_budget_mib):
        raise ValueError('case range or CPU/memory budget is invalid')
    for path, expected in ((manifest, manifest_sha256), (protocol, protocol_sha256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('frozen input hash mismatch')
    payload = json.loads(manifest.read_bytes())
    if not isinstance(payload.get('rows'), list) or end > len(payload['rows']):
        raise ValueError('case range exceeds manifest')
    if 'SLURM_CPUS_PER_TASK' in os.environ and cpu_limit > int(os.environ['SLURM_CPUS_PER_TASK']):
        raise ValueError('CPU limit exceeds scheduler allocation')
    if 'SLURM_MEM_PER_NODE' in os.environ and memory_budget_mib > int(os.environ['SLURM_MEM_PER_NODE']):
        raise ValueError('memory budget exceeds scheduler allocation')
    output_root.mkdir(parents=True, exist_ok=True)
    if any((output_root / f'case_{i:04d}').exists() for i in range(start, end)):
        raise FileExistsError('case output exists; reconcile before retry')
    control = output_root / f'packed_{start:04d}_{end:04d}'
    control.mkdir(exist_ok=False)
    env = dict(os.environ)
    env.update({name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})

    def run_one(index: int) -> dict:
        command = [sys.executable, '-m', 'phc2d.inverse.audit_perturb_refine',
                   '--manifest', str(manifest), '--manifest-sha256', manifest_sha256,
                   '--protocol', str(protocol), '--protocol-sha256', protocol_sha256,
                   '--case-index', str(index), '--output-root', str(output_root)]
        resource_file = control / f'case_{index:04d}.resources'
        if worker_resource_report:
            command = ['/usr/bin/time', '--quiet', '-o', str(resource_file), '-f',
                       '%e|%U|%S|%M|%x', *command]
        with (control / f'case_{index:04d}.log').open('xb') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
        record = {'case_index': index, 'returncode': result.returncode}
        if worker_resource_report:
            try:
                elapsed, user, system, rss_kib, reported_exit = resource_file.read_text().strip().split('|')
                elapsed, user, system = float(elapsed), float(user), float(system)
                rss_kib, reported_exit = int(rss_kib), int(reported_exit)
                if (not all(math.isfinite(x) and x >= 0 for x in (elapsed, user, system))
                        or rss_kib < 0 or reported_exit != result.returncode):
                    raise ValueError('invalid resource measurements')
            except (OSError, ValueError):
                record['resource_status'] = 'unavailable_or_invalid'
            else:
                record.update(resource_status='reported', elapsed_seconds=elapsed,
                              user_cpu_seconds=user, system_cpu_seconds=system,
                              max_rss_kib=rss_kib)
        return record

    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(executor.map(run_one, range(start, end)))
    result = {'manifest_sha256': manifest_sha256, 'protocol_sha256': protocol_sha256,
              'workers': workers, 'cpu_limit': cpu_limit, 'memory_budget_mib': memory_budget_mib,
              'worker_memory_mib': worker_memory_mib, 'setup_reserve_mib': setup_reserve_mib,
              'cases': outcomes, 'all_succeeded': all(item['returncode'] == 0 for item in outcomes)}
    if worker_resource_report:
        result['worker_resource_reporting'] = 'gnu_time_per_process_not_cgroup_peak'
        result['all_worker_resources_reported'] = all(item.get('resource_status') == 'reported' for item in outcomes)
    (control / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a bounded batch of inverse refinement cases')
    for name in ('manifest', 'manifest-sha256', 'protocol', 'protocol-sha256', 'output-root'):
        parser.add_argument('--' + name, required=True)
    for name in ('start', 'end', 'workers', 'cpu-limit', 'memory-budget-mib', 'worker-memory-mib', 'setup-reserve-mib'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--worker-resource-report', action='store_true')
    args = parser.parse_args()
    result = run_packed_refinement(manifest=Path(args.manifest), manifest_sha256=args.manifest_sha256,
        protocol=Path(args.protocol), protocol_sha256=args.protocol_sha256, output_root=Path(args.output_root),
        start=args.start, end=args.end, workers=args.workers, cpu_limit=args.cpu_limit,
        memory_budget_mib=args.memory_budget_mib, worker_memory_mib=args.worker_memory_mib,
        setup_reserve_mib=args.setup_reserve_mib, worker_resource_report=args.worker_resource_report)
    print(json.dumps(result))
    if not result['all_succeeded']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
