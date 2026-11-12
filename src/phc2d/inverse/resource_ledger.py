"""Track reserved and measured compute for inverse-design comparisons.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Mapping

from phc2d.hpc.slurm_accounting import parse_terminal_allocations


@dataclass(frozen=True)
class ResourceAmount:
    """Allocated CPU core-seconds and GPU-seconds, measured separately.

    Settlement should use allocated cores or GPUs multiplied by elapsed job
    time from a documented scheduler source; utilization time is not the
    same quantity. Wall time and pretraining costs are tracked separately.
    """

    cpu_core_seconds: float
    gpu_seconds: float

    def __post_init__(self) -> None:
        if not all(isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
                   for x in (self.cpu_core_seconds, self.gpu_seconds)):
            raise ValueError('resource amounts must be finite and nonnegative')


@dataclass(frozen=True)
class ResourceRecord:
    task_id: str
    reserved: ResourceAmount
    actual: ResourceAmount | None = None
    source: str | None = None
    state: str = 'reserved'
    allocation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.task_id, str) or not self.task_id.strip() or not isinstance(self.reserved, ResourceAmount)
                or not isinstance(self.state, str) or self.state not in {'reserved', 'settled', 'released'}
                or (self.state == 'settled' and (not isinstance(self.actual, ResourceAmount)
                                                 or not isinstance(self.source, str) or not self.source.strip()))
                or (self.state == 'reserved' and (self.actual is not None or self.source is not None))
                or (self.state == 'released' and (self.actual is not None or not isinstance(self.source, str)
                                                 or not self.source.strip()))
                or not isinstance(self.allocation_ids, tuple)
                or any(not isinstance(value, str) or not value for value in self.allocation_ids)
                or len(set(self.allocation_ids)) != len(self.allocation_ids)
                or (self.state != 'settled' and self.allocation_ids)):
            raise ValueError('invalid resource record')


@dataclass(frozen=True)
class ResourceLedger:
    limits: ResourceAmount
    records: tuple[ResourceRecord, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.limits, ResourceAmount) or not isinstance(self.records, tuple)
                or not all(isinstance(r, ResourceRecord) for r in self.records)
                or len({r.task_id for r in self.records}) != len(self.records)):
            raise ValueError('invalid resource ledger')
        allocations = [identity for record in self.records for identity in record.allocation_ids]
        if len(allocations) != len(set(allocations)):
            raise ValueError('scheduler allocation is already accounted')

    def totals(self) -> dict[str, ResourceAmount]:
        """Separate settled usage from outstanding reservations."""
        actual = [r.actual for r in self.records if r.state == 'settled']
        reserved = [r.reserved for r in self.records if r.state == 'reserved']
        def total(values: list[ResourceAmount]) -> ResourceAmount:
            return ResourceAmount(sum(x.cpu_core_seconds for x in values), sum(x.gpu_seconds for x in values))
        return {'actual': total(actual), 'reserved': total(reserved)}

    def status(self) -> str:
        totals = self.totals()
        if (totals['actual'].cpu_core_seconds + totals['reserved'].cpu_core_seconds > self.limits.cpu_core_seconds
                or totals['actual'].gpu_seconds + totals['reserved'].gpu_seconds > self.limits.gpu_seconds):
            return 'budget_exceeded'
        return 'within_budget'


def restore_resource_ledger(payload: Mapping[str, Any]) -> ResourceLedger:
    """Restore a serialized ledger without discarding reservations or overruns.

    Scheduler allocation identities remain subject to the same duplicate
    checks as live settlement. Legacy records without allocation IDs remain
    readable, but cannot retroactively establish which jobs they accounted for.
    """
    if not isinstance(payload, Mapping) or not isinstance(payload.get('limits'), Mapping):
        raise ValueError('invalid saved resource ledger')
    records = payload.get('records')
    if not isinstance(records, (list, tuple)):
        raise ValueError('saved resource records must be a sequence')

    def amount(raw: Any) -> ResourceAmount:
        if not isinstance(raw, Mapping) or 'cpu_core_seconds' not in raw or 'gpu_seconds' not in raw:
            raise ValueError('saved resource amount is incomplete')
        return ResourceAmount(raw['cpu_core_seconds'], raw['gpu_seconds'])

    restored = []
    for row in records:
        if not isinstance(row, Mapping) or not all(key in row for key in ('task_id', 'reserved', 'actual', 'source', 'state')):
            raise ValueError('saved resource record is incomplete')
        identities = row.get('allocation_ids', ())
        if not isinstance(identities, (list, tuple)):
            raise ValueError('saved allocation identities must be a sequence')
        restored.append(ResourceRecord(task_id=row['task_id'], reserved=amount(row['reserved']),
            actual=None if row['actual'] is None else amount(row['actual']), source=row['source'],
            state=row['state'], allocation_ids=tuple(identities)))
    return ResourceLedger(limits=amount(payload['limits']), records=tuple(restored))


def reserve_resources(ledger: ResourceLedger, *, task_id: str, amount: ResourceAmount) -> ResourceLedger:
    """Reserve an upper bound before dispatch; refuse duplicates and over-budget work."""
    if not task_id or any(r.task_id == task_id for r in ledger.records):
        raise ValueError('task identity must be nonempty and unique')
    totals = ledger.totals()
    if (totals['actual'].cpu_core_seconds + totals['reserved'].cpu_core_seconds + amount.cpu_core_seconds
            > ledger.limits.cpu_core_seconds
            or totals['actual'].gpu_seconds + totals['reserved'].gpu_seconds + amount.gpu_seconds
            > ledger.limits.gpu_seconds):
        raise ValueError('resource reservation exceeds budget')
    return replace(ledger, records=ledger.records + (ResourceRecord(task_id, amount),))


def settle_resources(ledger: ResourceLedger, *, task_id: str, actual: ResourceAmount,
                     source: str) -> ResourceLedger:
    """Record measured use, including actual overruns, from an identified source."""
    matches = [i for i, row in enumerate(ledger.records) if row.task_id == task_id]
    if len(matches) != 1 or ledger.records[matches[0]].state != 'reserved' or not source:
        raise ValueError('one reserved task and a measurement source are required')
    index = matches[0]
    records = list(ledger.records)
    records[index] = replace(records[index], actual=actual, source=source, state='settled')
    return replace(ledger, records=tuple(records))


def settle_slurm_resources(ledger: ResourceLedger, *, task_id: str, cluster_id: str,
                           accounting_bytes: bytes, expected_gpus: Mapping[str, int]) -> ResourceLedger:
    """Settle terminal Slurm allocations once, including failed-job costs.

    Job IDs are scoped by cluster; the immutable source digest and explicit
    allocation IDs make duplicate Slurm settlements detectable within this
    ledger. Caller retains the raw accounting snapshot as run evidence and
    verifies that no allocation can be requeued before settlement.
    """
    if not isinstance(cluster_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', cluster_id):
        raise ValueError('a stable cluster identity is required')
    parsed = parse_terminal_allocations(accounting_bytes, expected_gpus=expected_gpus)
    identities = tuple(f'{cluster_id}:{record.job_id}' for record in parsed['allocations'])
    existing = {identity for record in ledger.records for identity in record.allocation_ids}
    if existing.intersection(identities):
        raise ValueError('scheduler allocation is already accounted')
    settled = settle_resources(ledger, task_id=task_id,
        actual=ResourceAmount(parsed['cpu_core_seconds'], parsed['gpu_seconds']),
        source=f"slurm:{cluster_id}:sha256:{parsed['source_sha256']}")
    records = list(settled.records)
    index = next(i for i, record in enumerate(records) if record.task_id == task_id)
    records[index] = replace(records[index], allocation_ids=identities)
    return replace(settled, records=tuple(records))


def settle_slurm_worker_group(
    ledger: ResourceLedger, *, task_id: str, cluster_id: str, job_id: str,
    accounting_bytes: bytes, expected_gpus: int, expected_workers: Mapping[str, str],
    worker_outcomes: Mapping[str, Mapping[str, Any]],
) -> tuple[ResourceLedger, dict[str, Any]]:
    """Settle one allocation once after checking every frozen worker identity.

    Keys are worker IDs; expected values are candidate SHA-256 identities.
    Outcomes contain candidate_sha256, integer or unknown (None) exit_code and
    result_sha256 (required on success; optional otherwise). Raw output validation, worker
    outcome authenticity, job-to-worker mapping and requeue status remain the
    caller's responsibility. This operation does not assign job cost to workers
    or judge numerical/physical success.
    """
    digest = re.compile(r'^[0-9a-f]{64}$')
    if (not isinstance(expected_workers, Mapping) or not expected_workers
            or not isinstance(worker_outcomes, Mapping) or set(worker_outcomes) != set(expected_workers)
            or any(not isinstance(worker, str) or not worker.strip() for worker in expected_workers)
            or any(not isinstance(value, str) or not digest.fullmatch(value) for value in expected_workers.values())
            or len(set(expected_workers.values())) != len(expected_workers)):
        raise ValueError('invalid or incomplete frozen worker roster')
    failed, unknown = [], []
    for worker, candidate in expected_workers.items():
        result = worker_outcomes[worker]
        if not isinstance(result, Mapping) or set(result) != {'candidate_sha256', 'exit_code', 'result_sha256'}:
            raise ValueError('invalid worker outcome fields')
        code, result_sha = result['exit_code'], result['result_sha256']
        if (result['candidate_sha256'] != candidate
                or (code is not None and (isinstance(code, bool) or not isinstance(code, int) or code < 0))
                or (result_sha is not None and (not isinstance(result_sha, str) or not digest.fullmatch(result_sha)))
                or (code == 0 and result_sha is None)):
            raise ValueError('worker identity, exit or result digest mismatch')
        if code is None:
            unknown.append(worker)
        elif code != 0:
            failed.append(worker)
    parsed = parse_terminal_allocations(accounting_bytes, expected_gpus={job_id: expected_gpus})
    settled = settle_slurm_resources(ledger, task_id=task_id, cluster_id=cluster_id,
                                     accounting_bytes=accounting_bytes, expected_gpus={job_id: expected_gpus})
    return settled, {'scope': 'worker_identity_and_single_allocation_only_not_physical_validation',
        'job_id': job_id, 'allocation_state': parsed['allocations'][0].state,
        'worker_count': len(expected_workers), 'failed_workers': failed, 'unknown_workers': unknown,
        'cpu_core_seconds': parsed['cpu_core_seconds'], 'gpu_seconds': parsed['gpu_seconds'],
        'accounting_sha256': parsed['source_sha256']}


def release_unstarted(ledger: ResourceLedger, *, task_id: str, confirmed_not_started: bool,
                      confirmation_source: str) -> ResourceLedger:
    """Release a reservation only after external confirmation that it never ran."""
    matches = [i for i, row in enumerate(ledger.records) if row.task_id == task_id]
    if (confirmed_not_started is not True or not confirmation_source or len(matches) != 1
            or ledger.records[matches[0]].state != 'reserved'):
        raise ValueError('an unstarted reserved task must be externally confirmed')
    records = list(ledger.records)
    records[matches[0]] = replace(records[matches[0]], state='released', source=confirmation_source)
    return replace(ledger, records=tuple(records))


def audit_slurm_ledger(ledger: ResourceLedger, *, accounting_by_task: Mapping[str, Mapping[str, Any]],
                       require_settled: bool = False) -> dict[str, Any]:
    """Recompute settled Slurm allocations and verify one closed trial budget.

    Every settled task needs its raw ``sacct`` bytes, cluster identity and
    explicit expected GPU counts. Pending or released reservations fail
    closed: an unstarted release needs separate external evidence. This audit
    does not prove that an allocation cannot be requeued, that accounting is
    authentic, or that these jobs cover every piece of trial work; callers
    must establish those facts and account separately for excluded costs.
    """
    if (not isinstance(ledger, ResourceLedger) or not isinstance(accounting_by_task, Mapping)
            or not isinstance(require_settled, bool)):
        raise ValueError('resource ledger and accounting evidence are required')
    if any(record.state != 'settled' for record in ledger.records):
        raise ValueError('trial has pending or released reservations')
    if require_settled and not ledger.records:
        raise ValueError('attempted trial has no settled accounting')
    if set(accounting_by_task) != {record.task_id for record in ledger.records}:
        raise ValueError('accounting evidence does not match settled tasks')
    identities: list[str] = []
    for record in ledger.records:
        evidence = accounting_by_task[record.task_id]
        if not isinstance(evidence, Mapping):
            raise ValueError('invalid task accounting evidence')
        cluster, raw, expected = evidence.get('cluster_id'), evidence.get('raw'), evidence.get('expected_gpus')
        if (not isinstance(cluster, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', cluster)
                or not isinstance(raw, bytes) or not isinstance(expected, Mapping)):
            raise ValueError('cluster, raw accounting and expected GPUs are required')
        parsed = parse_terminal_allocations(raw, expected_gpus=expected)
        expected_ids = {f'{cluster}:{entry.job_id}' for entry in parsed['allocations']}
        if (set(record.allocation_ids) != expected_ids
                or record.source != f"slurm:{cluster}:sha256:{parsed['source_sha256']}"
                or record.actual != ResourceAmount(parsed['cpu_core_seconds'], parsed['gpu_seconds'])):
            raise ValueError('settled ledger does not match raw scheduler accounting')
        identities.extend(record.allocation_ids)
    if len(identities) != len(set(identities)):
        raise ValueError('scheduler allocation is already accounted')
    if ledger.status() != 'within_budget':
        raise ValueError('trial actual resource usage exceeds its budget')
    total = ledger.totals()['actual']
    return {'cpu_core_seconds': total.cpu_core_seconds, 'gpu_seconds': total.gpu_seconds,
            'allocation_ids': tuple(identities), 'settled_tasks': len(ledger.records),
            'scope': 'recomputed_slurm_records_caller_verifies_completeness_and_no_requeue'}
