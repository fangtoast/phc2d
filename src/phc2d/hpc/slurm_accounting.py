"""Verify terminal Slurm allocation records without counting job steps twice.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Mapping


_TERMINAL_STATES = frozenset({'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT',
                              'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL',
                              'DEADLINE', 'REVOKED'})
_JOB_ID = re.compile(r'^[0-9]+(?:_[0-9]+)?$')


@dataclass(frozen=True)
class AllocationCost:
    """One allocation's scheduler-reported elapsed time and resources."""

    job_id: str
    state: str
    elapsed_seconds: int
    allocated_cpus: int
    allocated_gpus: int

    @property
    def cpu_core_seconds(self) -> int:
        return self.elapsed_seconds * self.allocated_cpus

    @property
    def gpu_seconds(self) -> int:
        return self.elapsed_seconds * self.allocated_gpus


def _gpu_count(tres: str, *, allocated_cpus: int) -> int:
    fields: dict[str, str] = {}
    for item in tres.split(','):
        if '=' not in item:
            raise ValueError('malformed allocated TRES')
        key, value = item.split('=', 1)
        if not key or key in fields:
            raise ValueError('duplicate or empty allocated TRES')
        fields[key] = value
    if not fields.get('cpu', '').isdecimal() or int(fields['cpu']) != allocated_cpus:
        raise ValueError('CPU TRES and allocated CPU count disagree')
    gpu_fields = {key: value for key, value in fields.items() if key == 'gres/gpu' or key.startswith('gres/gpu:')}
    if any(not value.isdecimal() for value in gpu_fields.values()):
        raise ValueError('GPU allocation must be a nonnegative integer')
    typed = sum(int(value) for key, value in gpu_fields.items() if key != 'gres/gpu')
    if 'gres/gpu' in gpu_fields:
        total = int(gpu_fields['gres/gpu'])
        if len(gpu_fields) > 1 and total != typed:
            raise ValueError('aggregate and typed GPU allocations disagree')
        return total
    return typed


def parse_terminal_allocations(raw: bytes, *, expected_gpus: Mapping[str, int]) -> dict:
    """Parse ``sacct -P -n`` JobID,State,ElapsedRaw,AllocCPUS,AllocTRES.

    Only exact, explicitly expected allocation IDs are admitted. Array members
    are distinct IDs; scheduler steps (``.batch``, ``.extern``, etc.) are
    ignored. A running, missing, duplicate, or unrequested allocation fails
    closed. Explicit expected GPU counts distinguish a CPU allocation from a
    missing GPU TRES field. This records allocated resource time, not energy,
    GPU utilization, wall-clock critical path, or queue time. The caller must
    also establish that the terminal snapshot cannot later be requeued before
    treating it as final accounting.
    """
    if not expected_gpus or any(not isinstance(job, str) or not _JOB_ID.fullmatch(job) or isinstance(count, bool)
                                or not isinstance(count, int) or count < 0
                                for job, count in expected_gpus.items()):
        raise ValueError('expected allocations and GPU counts must be explicit')
    records: dict[str, AllocationCost] = {}
    for line in raw.decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        columns = line.rstrip('\r').split('|')
        if len(columns) != 5:
            raise ValueError('expected exactly five sacct columns')
        job, state, elapsed, cpus, tres = (value.strip() for value in columns)
        if '.' in job:
            parent = job.split('.', 1)[0]
            if parent not in expected_gpus:
                raise ValueError('unrequested scheduler job step')
            continue
        if job not in expected_gpus:
            raise ValueError('unrequested scheduler allocation')
        if job in records:
            raise ValueError('duplicate scheduler allocation')
        normalized_state = state.split()[0].rstrip('+') if state else ''
        if normalized_state not in _TERMINAL_STATES:
            raise ValueError('allocation is not terminal')
        if not elapsed.isdecimal() or not cpus.isdecimal() or int(cpus) <= 0:
            raise ValueError('allocated CPU count or elapsed time unavailable')
        gpu_count = _gpu_count(tres, allocated_cpus=int(cpus))
        if gpu_count != expected_gpus[job]:
            raise ValueError('GPU allocation does not match expected count')
        records[job] = AllocationCost(job, normalized_state, int(elapsed), int(cpus), gpu_count)
    if records.keys() != expected_gpus.keys():
        raise ValueError('missing expected scheduler allocation')
    ordered = [records[job] for job in expected_gpus]
    return {'source_sha256': hashlib.sha256(raw).hexdigest(), 'allocations': ordered,
            'cpu_core_seconds': sum(row.cpu_core_seconds for row in ordered),
            'gpu_seconds': sum(row.gpu_seconds for row in ordered)}
