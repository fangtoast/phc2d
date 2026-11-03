"""Pure multi-worker feedback dispatch with one allocation per batch.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Mapping

from phc2d.inverse.feedback_handoff import consume_robust_worker_group
from phc2d.inverse.feedback_state import FeedbackState, propose_feedback, restore_feedback
from phc2d.inverse.resource_ledger import (ResourceAmount, ResourceLedger, release_unstarted, reserve_resources,
                                           restore_resource_ledger, settle_slurm_worker_group)


@dataclass(frozen=True)
class WorkerBatch:
    """One scheduler allocation with a frozen worker-to-lane/candidate roster."""
    task_id: str
    cluster_id: str
    workers: Mapping[str, str]
    candidates: Mapping[str, str]
    phase: str
    job_id: str | None = None
    confirmation_source: str | None = None
    outcomes: Mapping[str, Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class FeedbackBatchState:
    """Concurrent batches sharing one ledger; caller supplies durable control.

    One candidate consumes one attempt when reserved, even if later confirmed
    unstarted. All external actions require a single writer, trustworthy
    persisted state, and independent scheduler/evidence verification.
    """
    lanes: Mapping[str, FeedbackState]
    batches: Mapping[str, WorkerBatch]
    ledger: ResourceLedger
    max_inflight: int
    max_attempts: int
    attempts: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.lanes, Mapping) or not self.lanes or not isinstance(self.batches, Mapping)
                or not isinstance(self.ledger, ResourceLedger)
                or any(not isinstance(k, str) or not k or not isinstance(v, FeedbackState) for k, v in self.lanes.items())
                or any(isinstance(x, bool) or not isinstance(x, int) for x in (self.max_inflight, self.max_attempts, self.attempts))
                or self.max_inflight <= 0 or self.max_attempts <= 0 or not 0 <= self.attempts <= self.max_attempts
                or len(self.batches) > self.max_inflight):
            raise ValueError('invalid batch campaign state or limits')
        active_lanes, active_candidates, task_ids, jobs = [], [], [], []
        for name, batch in self.batches.items():
            if (not isinstance(name, str) or not name or not isinstance(batch, WorkerBatch)
                    or not isinstance(batch.task_id, str) or not batch.task_id
                    or not isinstance(batch.cluster_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', batch.cluster_id)
                    or not isinstance(batch.workers, Mapping) or not batch.workers
                    or not isinstance(batch.candidates, Mapping) or set(batch.workers) != set(batch.candidates)
                    or any(not isinstance(w, str) or not w or lane not in self.lanes for w, lane in batch.workers.items())
                    or any(not isinstance(c, str) or not re.fullmatch(r'[0-9a-f]{64}', c) for c in batch.candidates.values())
                    or len(set(batch.workers.values())) != len(batch.workers)
                    or len(set(batch.candidates.values())) != len(batch.candidates)
                    or batch.phase not in {'prepared', 'submission_unknown', 'submitted', 'awaiting_accounting'}):
                raise ValueError('invalid worker batch')
            if batch.phase == 'prepared':
                if batch.job_id is not None or batch.confirmation_source is not None or batch.outcomes is not None:
                    raise ValueError('prepared batch cannot claim submission')
            elif batch.phase == 'submission_unknown':
                if batch.job_id is not None or not batch.confirmation_source or batch.outcomes is not None:
                    raise ValueError('ambiguous batch needs a reason and no job')
            else:
                if (not isinstance(batch.job_id, str) or not re.fullmatch(r'[0-9]+(?:_[0-9]+)?', batch.job_id)
                        or not isinstance(batch.confirmation_source, str) or not batch.confirmation_source.strip()):
                    raise ValueError('submitted batch needs a confirmed job')
                jobs.append((batch.cluster_id, batch.job_id))
                if (batch.phase == 'submitted') != (batch.outcomes is None):
                    raise ValueError('batch outcomes differ from phase')
            if batch.outcomes is not None and set(batch.outcomes) != set(batch.workers):
                raise ValueError('batch outcome roster mismatch')
            for worker, lane in batch.workers.items():
                state = self.lanes[lane]
                if batch.phase == 'awaiting_accounting':
                    if (state.pending is not None or not state.history
                            or state.history[-1]['parameter_hash'] != batch.candidates[worker]
                            or not isinstance(batch.outcomes[worker], Mapping)
                            or set(batch.outcomes[worker]) != {'candidate_sha256', 'exit_code', 'result_sha256'}
                            or batch.outcomes[worker].get('candidate_sha256') != batch.candidates[worker]
                            or isinstance(batch.outcomes[worker].get('exit_code'), bool)
                            or not isinstance(batch.outcomes[worker].get('exit_code'), int)
                            or batch.outcomes[worker]['exit_code'] < 0
                            or ((state.history[-1]['status'] == 'solver_failure') !=
                                (batch.outcomes[worker]['exit_code'] != 0))
                            or (batch.outcomes[worker]['exit_code'] == 0 and
                                (not isinstance(batch.outcomes[worker]['result_sha256'], str) or
                                 not re.fullmatch(r'[0-9a-f]{64}', batch.outcomes[worker]['result_sha256'])))
                            or (batch.outcomes[worker]['exit_code'] != 0 and batch.outcomes[worker]['result_sha256'] is not None)):
                        raise ValueError('settled worker and lane differ')
                elif state.pending is None or state.pending.parameter_hash != batch.candidates[worker]:
                    raise ValueError('pending worker and lane differ')
            active_lanes.extend(batch.workers.values())
            active_candidates.extend(batch.candidates.values())
            task_ids.append(batch.task_id)
        reserved = {record.task_id for record in self.ledger.records if record.state == 'reserved'}
        accounted = {identity for record in self.ledger.records for identity in record.allocation_ids}
        if (len(active_lanes) != len(set(active_lanes)) or len(active_candidates) != len(set(active_candidates))
                or len(task_ids) != len(set(task_ids))
                or set(task_ids) != reserved or len(jobs) != len(set(jobs))
                or any(f'{cluster}:{job}' in accounted for cluster, job in jobs)
                or any(state.pending is not None and name not in active_lanes for name, state in self.lanes.items())
                or self.attempts < sum(len(state.history) for state in self.lanes.values())
                   + sum(len(batch.workers) for batch in self.batches.values() if batch.phase != 'awaiting_accounting')):
            raise ValueError('duplicate or unowned batch state')


def prepare_worker_batch(state: FeedbackBatchState, *, batch_id: str, task_id: str, cluster_id: str,
                         proposals: Mapping[str, Mapping[str, Any]], workers: Mapping[str, str],
                         reservation: ResourceAmount) -> FeedbackBatchState:
    """Reserve one job and each worker attempt before any external submission."""
    if (not isinstance(batch_id, str) or not batch_id or batch_id in state.batches
            or not isinstance(proposals, Mapping) or not isinstance(workers, Mapping) or not workers
            or set(proposals) != set(workers) or len(set(workers.values())) != len(workers)
            or len(state.batches) >= state.max_inflight or state.attempts + len(workers) > state.max_attempts
            or state.ledger.status() != 'within_budget'):
        raise ValueError('invalid batch, concurrency or attempt budget')
    lanes, candidates = dict(state.lanes), {}
    for worker, lane in workers.items():
        if lane not in lanes:
            raise ValueError('unknown worker lane')
        lanes[lane] = propose_feedback(lanes[lane], **dict(proposals[worker]))
        candidates[worker] = lanes[lane].pending.parameter_hash
    if len(set(candidates.values())) != len(candidates):
        raise ValueError('duplicate batch candidate')
    ledger = reserve_resources(state.ledger, task_id=task_id, amount=reservation)
    batches = dict(state.batches)
    batches[batch_id] = WorkerBatch(task_id, cluster_id, dict(workers), candidates, 'prepared')
    return FeedbackBatchState(lanes, batches, ledger, state.max_inflight, state.max_attempts,
                              state.attempts + len(workers))


def mark_batch_submission_unknown(state: FeedbackBatchState, *, batch_id: str, reason: str) -> FeedbackBatchState:
    batch = state.batches[batch_id]
    if batch.phase != 'prepared' or not isinstance(reason, str) or not reason.strip():
        raise ValueError('prepared batch and reason required')
    batches = dict(state.batches)
    batches[batch_id] = replace(batch, phase='submission_unknown', confirmation_source=reason)
    return replace(state, batches=batches)


def confirm_worker_batch(state: FeedbackBatchState, *, batch_id: str, job_id: str,
                         confirmation_source: str) -> FeedbackBatchState:
    batch = state.batches[batch_id]
    if (batch.phase != 'submission_unknown' or not isinstance(job_id, str) or not job_id
            or not isinstance(confirmation_source, str) or not confirmation_source.strip()):
        raise ValueError('ambiguous batch and confirmed scheduler identity required')
    batches = dict(state.batches)
    batches[batch_id] = replace(batch, phase='submitted', job_id=job_id, confirmation_source=confirmation_source)
    return replace(state, batches=batches)


def release_unstarted_worker_batch(state: FeedbackBatchState, *, batch_id: str, confirmed_not_started: bool,
                                   confirmed_no_future_execution: bool, confirmation_source: str) -> FeedbackBatchState:
    """Release an ambiguous job only when it never ran and cannot run later."""
    batch = state.batches[batch_id]
    if batch.phase != 'submission_unknown' or confirmed_no_future_execution is not True:
        raise ValueError('ambiguous batch and no-future-execution confirmation required')
    ledger = release_unstarted(state.ledger, task_id=batch.task_id, confirmed_not_started=confirmed_not_started,
                               confirmation_source=confirmation_source)
    lanes = dict(state.lanes)
    for lane in batch.workers.values():
        lanes[lane] = replace(lanes[lane], pending=None)
    batches = dict(state.batches)
    del batches[batch_id]
    return replace(state, lanes=lanes, batches=batches, ledger=ledger)


def reconcile_worker_batch(state: FeedbackBatchState, *, batch_id: str,
                           results_by_worker: Mapping[str, Mapping[str, Any]]) -> FeedbackBatchState:
    """Consume a complete batch only after its external evidence is authenticated."""
    batch = state.batches[batch_id]
    if batch.phase != 'submitted':
        raise ValueError('confirmed submitted batch required')
    pending = {worker: state.lanes[lane] for worker, lane in batch.workers.items()}
    updated, outcomes = consume_robust_worker_group(pending, results_by_worker)
    lanes = dict(state.lanes)
    for worker, lane in batch.workers.items():
        lanes[lane] = updated[worker]
    batches = dict(state.batches)
    batches[batch_id] = replace(batch, phase='awaiting_accounting', outcomes=outcomes)
    return replace(state, lanes=lanes, batches=batches)


def settle_worker_batch(state: FeedbackBatchState, *, batch_id: str,
                        accounting_bytes: bytes, expected_gpus: int) -> tuple[FeedbackBatchState, dict[str, Any]]:
    """Account once for the allocation after every worker has a recorded outcome."""
    batch = state.batches[batch_id]
    if batch.phase != 'awaiting_accounting':
        raise ValueError('complete worker outcomes required before settlement')
    ledger, audit = settle_slurm_worker_group(state.ledger, task_id=batch.task_id, cluster_id=batch.cluster_id,
        job_id=batch.job_id, accounting_bytes=accounting_bytes, expected_gpus=expected_gpus,
        expected_workers=batch.candidates, worker_outcomes=batch.outcomes)
    batches = dict(state.batches)
    del batches[batch_id]
    return replace(state, ledger=ledger, batches=batches), audit


def restore_feedback_batch_state(payload: Mapping[str, Any]) -> FeedbackBatchState:
    """Restore batch state; independent journal and trusted evidence are required."""
    lanes = {name: restore_feedback(item) for name, item in payload['lanes'].items()}
    batches = {name: WorkerBatch(**item) for name, item in payload['batches'].items()}
    return FeedbackBatchState(lanes, batches, restore_resource_ledger(payload['ledger']),
                              payload['max_inflight'], payload['max_attempts'], payload['attempts'])
