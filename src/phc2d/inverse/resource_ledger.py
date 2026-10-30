"""Track reserved and measured compute for inverse-design comparisons.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


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
        if not all(math.isfinite(x) and x >= 0 for x in (self.cpu_core_seconds, self.gpu_seconds)):
            raise ValueError('resource amounts must be finite and nonnegative')


@dataclass(frozen=True)
class ResourceRecord:
    task_id: str
    reserved: ResourceAmount
    actual: ResourceAmount | None = None
    source: str | None = None
    state: str = 'reserved'

    def __post_init__(self) -> None:
        if (not self.task_id or not isinstance(self.reserved, ResourceAmount)
                or self.state not in {'reserved', 'settled', 'released'}
                or (self.state == 'settled' and (not isinstance(self.actual, ResourceAmount) or not self.source))
                or (self.state == 'reserved' and (self.actual is not None or self.source is not None))
                or (self.state == 'released' and (self.actual is not None or not self.source))):
            raise ValueError('invalid resource record')


@dataclass(frozen=True)
class ResourceLedger:
    limits: ResourceAmount
    records: tuple[ResourceRecord, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.limits, ResourceAmount) or not all(isinstance(r, ResourceRecord) for r in self.records)
                or len({r.task_id for r in self.records}) != len(self.records)):
            raise ValueError('invalid resource ledger')

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
