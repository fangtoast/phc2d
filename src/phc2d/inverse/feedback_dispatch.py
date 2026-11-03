"""Coordinate one externally dispatched feedback attempt and its budget.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

from phc2d.hpc.slurm_accounting import parse_terminal_allocations
from phc2d.inverse.feedback_handoff import consume_robust_solver_result
from phc2d.inverse.feedback_state import FeedbackState, finish_feedback, propose_feedback, restore_feedback
from phc2d.inverse.resource_ledger import (ResourceAmount, ResourceLedger, release_unstarted,
                                           reserve_resources, restore_resource_ledger, settle_slurm_resources)


_JOB = re.compile(r'^[0-9]+(?:_[0-9]+)?$')
_CLUSTER = re.compile(r'^[A-Za-z0-9_.-]+$')
_FAILURES = frozenset({'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
                       'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'})


@dataclass(frozen=True)
class Dispatch:
    task_id: str
    cluster_id: str
    candidate_hash: str
    phase: str
    job_id: str | None = None
    confirmation_source: str | None = None
    result_sha256: str | None = None
    terminal_failure: str | None = None


@dataclass(frozen=True)
class FeedbackDispatchState:
    """Pure state; callers persist each transition before external actions.

    This does not invoke Slurm, lock files, or verify prior journal chains.
    A single controller must validate the initial incumbent and existing
    journals. Persist submission_unknown before calling the scheduler and
    reconcile ambiguous submissions rather than resend them.
    """

    feedback: FeedbackState
    ledger: ResourceLedger
    dispatch: Dispatch | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.feedback, FeedbackState) or not isinstance(self.ledger, ResourceLedger):
            raise ValueError('feedback and resource ledger are required')
        item = self.dispatch
        if item is None:
            if self.feedback.pending is not None:
                raise ValueError('pending feedback requires a dispatch')
            return
        if (not isinstance(item, Dispatch) or not isinstance(item.task_id, str) or not item.task_id.strip()
                or not isinstance(item.cluster_id, str) or not _CLUSTER.fullmatch(item.cluster_id)
                or not isinstance(item.candidate_hash, str) or not item.candidate_hash
                or not isinstance(item.phase, str)
                or item.phase not in {'prepared', 'submission_unknown', 'submitted', 'awaiting_accounting'}):
            raise ValueError('invalid dispatch identity or phase')
        records = [record for record in self.ledger.records if record.task_id == item.task_id]
        if len(records) != 1 or records[0].state != 'reserved':
            raise ValueError('dispatch requires one reserved ledger task')
        if item.phase in {'prepared', 'submission_unknown'}:
            if item.job_id is not None or item.result_sha256 is not None or item.terminal_failure is not None:
                raise ValueError('unconfirmed dispatch cannot claim a job or result')
            if item.phase == 'prepared' and item.confirmation_source is not None:
                raise ValueError('prepared dispatch cannot claim a confirmation')
            if item.phase == 'submission_unknown' and (not isinstance(item.confirmation_source, str)
                                                        or not item.confirmation_source.strip()):
                raise ValueError('ambiguous submission needs a reason')
        else:
            if (not isinstance(item.job_id, str) or not _JOB.fullmatch(item.job_id)
                    or not isinstance(item.confirmation_source, str) or not item.confirmation_source.strip()):
                raise ValueError('submitted dispatch needs a confirmed scheduler identity')
        if item.phase != 'awaiting_accounting':
            if (self.feedback.pending is None or self.feedback.pending.parameter_hash != item.candidate_hash
                    or item.result_sha256 is not None or item.terminal_failure is not None):
                raise ValueError('dispatch and pending candidate differ')
        else:
            if (self.feedback.pending is not None or not self.feedback.history
                    or self.feedback.history[-1]['parameter_hash'] != item.candidate_hash):
                raise ValueError('completed attempt and feedback history differ')
            if (item.result_sha256 is None) == (item.terminal_failure is None):
                raise ValueError('record exactly one solver result or terminal failure')
            if item.result_sha256 is not None and (not isinstance(item.result_sha256, str)
                                                    or not re.fullmatch(r'[0-9a-f]{64}', item.result_sha256)):
                raise ValueError('invalid result digest')
            if item.terminal_failure is not None and (not isinstance(item.terminal_failure, str)
                                                       or item.terminal_failure not in _FAILURES):
                raise ValueError('invalid terminal failure')
            if (self.feedback.history[-1]['status'] == 'solver_failure') != (item.terminal_failure is not None):
                raise ValueError('terminal failure and feedback history differ')


def begin_dispatch(state: FeedbackDispatchState, *, proposal: Mapping[str, Any], task_id: str,
                   cluster_id: str, reservation: ResourceAmount) -> FeedbackDispatchState:
    """Reserve resources and candidate together before any scheduler call."""
    if state.dispatch is not None or state.ledger.status() != 'within_budget':
        raise ValueError('previous dispatch or budget must be reconciled')
    ledger = reserve_resources(state.ledger, task_id=task_id, amount=reservation)
    feedback = propose_feedback(state.feedback, **proposal)
    dispatch = Dispatch(task_id, cluster_id, feedback.pending.parameter_hash, 'prepared')
    return FeedbackDispatchState(feedback, ledger, dispatch)


def mark_submission_unknown(state: FeedbackDispatchState, *, reason: str) -> FeedbackDispatchState:
    """Persist before calling the scheduler; reconcile rather than resubmit."""
    item = state.dispatch
    if item is None or item.phase != 'prepared' or not isinstance(reason, str) or not reason.strip():
        raise ValueError('prepared dispatch and ambiguity reason required')
    return replace(state, dispatch=replace(item, phase='submission_unknown', confirmation_source=reason))


def confirm_submission(state: FeedbackDispatchState, *, job_id: str,
                       confirmation_source: str) -> FeedbackDispatchState:
    """Record a scheduler ID only after an external submission is confirmed."""
    item = state.dispatch
    if item is None or item.phase != 'submission_unknown':
        raise ValueError('dispatch is not awaiting submission confirmation')
    if not isinstance(confirmation_source, str) or not confirmation_source.strip():
        raise ValueError('submission confirmation source required')
    if f'{item.cluster_id}:{job_id}' in {identity for record in state.ledger.records for identity in record.allocation_ids}:
        raise ValueError('scheduler allocation is already accounted')
    return replace(state, dispatch=replace(item, phase='submitted', job_id=job_id,
                                           confirmation_source=confirmation_source))


def resolve_unstarted(state: FeedbackDispatchState, *, confirmed_not_started: bool,
                      confirmed_no_future_execution: bool,
                      confirmation_source: str) -> FeedbackDispatchState:
    """Release only when no execution has occurred or can occur in future."""
    item = state.dispatch
    if item is None or item.phase != 'submission_unknown' or confirmed_no_future_execution is not True:
        raise ValueError('only an ambiguous submission can be released')
    ledger = release_unstarted(state.ledger, task_id=item.task_id,
                               confirmed_not_started=confirmed_not_started,
                               confirmation_source=confirmation_source)
    return FeedbackDispatchState(replace(state.feedback, pending=None), ledger)


def record_robust_result(state: FeedbackDispatchState, *, manifest_bytes: bytes,
                         manifest_sha256: str, protocol_bytes: bytes, protocol_sha256: str,
                         result_bytes: bytes) -> FeedbackDispatchState:
    """Recompute a decision after persisting artifacts and checking job ownership."""
    item = state.dispatch
    if item is None or item.phase != 'submitted':
        raise ValueError('confirmed submitted dispatch required')
    feedback = consume_robust_solver_result(state.feedback, manifest_bytes=manifest_bytes,
        expected_manifest_sha256=manifest_sha256, protocol_bytes=protocol_bytes,
        protocol_sha256=protocol_sha256, result=json.loads(result_bytes))
    return FeedbackDispatchState(feedback, state.ledger,
        replace(item, phase='awaiting_accounting', result_sha256=hashlib.sha256(result_bytes).hexdigest()))


def record_terminal_failure(state: FeedbackDispatchState, *, failure: str) -> FeedbackDispatchState:
    """Consume a confirmed failed solver attempt; its resources remain reserved."""
    item = state.dispatch
    if item is None or item.phase != 'submitted' or not isinstance(failure, str) or failure not in _FAILURES:
        raise ValueError('submitted dispatch and terminal failure required')
    feedback = finish_feedback(state.feedback,
        failure_reason=f'slurm:{item.cluster_id}:{item.job_id}:{failure}')
    return FeedbackDispatchState(feedback, state.ledger,
        replace(item, phase='awaiting_accounting', terminal_failure=failure))


def settle_dispatch(state: FeedbackDispatchState, *, accounting_bytes: bytes,
                    allocated_gpus: int) -> FeedbackDispatchState:
    """Settle the exact allocation after caller confirms it cannot be requeued."""
    item = state.dispatch
    if item is None or item.phase != 'awaiting_accounting' or item.job_id is None:
        raise ValueError('recorded result and confirmed job required')
    expected = {item.job_id: allocated_gpus}
    parsed = parse_terminal_allocations(accounting_bytes, expected_gpus=expected)
    terminal = parsed['allocations'][0].state
    if terminal != (item.terminal_failure or 'COMPLETED'):
        raise ValueError('scheduler status does not match recorded solver outcome')
    ledger = settle_slurm_resources(state.ledger, task_id=item.task_id, cluster_id=item.cluster_id,
                                    accounting_bytes=accounting_bytes, expected_gpus=expected)
    return FeedbackDispatchState(state.feedback, ledger)


def restore_feedback_dispatch(payload: Mapping[str, Any]) -> FeedbackDispatchState:
    """Restore a persisted dispatch; verify journal evidence separately."""
    if not isinstance(payload, Mapping) or not all(key in payload for key in ('feedback', 'ledger', 'dispatch')):
        raise ValueError('invalid saved feedback dispatch')
    raw = payload['dispatch']
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError('invalid saved dispatch')
    return FeedbackDispatchState(restore_feedback(payload['feedback']), restore_resource_ledger(payload['ledger']),
                                 None if raw is None else Dispatch(**raw))


@dataclass(frozen=True)
class FeedbackCampaignState:
    """Independent feedback lanes sharing one conservative allocation ledger.

    Each dispatch covers one candidate. A single external controller must
    persist transitions and authenticate scheduler records; this class does not
    provide a lock, a journal, or scheduler access.
    """
    lanes: Mapping[str, FeedbackState]
    dispatches: Mapping[str, Dispatch]
    ledger: ResourceLedger
    max_inflight: int
    max_attempts: int

    def __post_init__(self) -> None:
        if (not isinstance(self.lanes, Mapping) or not self.lanes or not isinstance(self.dispatches, Mapping)
                or not isinstance(self.ledger, ResourceLedger) or isinstance(self.max_inflight, bool)
                or not isinstance(self.max_inflight, int) or self.max_inflight <= 0
                or isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int)
                or self.max_attempts <= 0 or len(self.dispatches) > self.max_inflight
                or not set(self.dispatches).issubset(self.lanes)
                or any(not isinstance(k, str) or not k for k in self.lanes)):
            raise ValueError('invalid campaign lanes or limits')
        for lane, feedback in self.lanes.items():
            FeedbackDispatchState(feedback, self.ledger, self.dispatches.get(lane))
        task_ids = [item.task_id for item in self.dispatches.values()]
        jobs = [(item.cluster_id, item.job_id) for item in self.dispatches.values() if item.job_id is not None]
        reserved = {row.task_id for row in self.ledger.records if row.state == 'reserved'}
        accounted = {identity for row in self.ledger.records for identity in row.allocation_ids}
        if (len(task_ids) != len(set(task_ids)) or set(task_ids) != reserved
                or len(jobs) != len(set(jobs)) or any(f'{cluster}:{job}' in accounted for cluster, job in jobs)
                or len(self.ledger.records) > self.max_attempts):
            raise ValueError('duplicate or unowned campaign reservations or jobs')


def apply_campaign_event(state: FeedbackCampaignState, *, lane: str, operation: str,
                         payload: Mapping[str, Any]) -> FeedbackCampaignState:
    """Apply one validated lane transition to the shared ledger, without I/O.

    Persist the returned state before external actions. Restore and verify it
    under single-writer control; ambiguous submissions retain their reservation.
    Released reservations count towards the conservative attempt cap.
    """
    operations = {'begin': begin_dispatch, 'submission_unknown': mark_submission_unknown,
                  'confirm': confirm_submission, 'release': resolve_unstarted,
                  'result': record_robust_result, 'failure': record_terminal_failure,
                  'settle': settle_dispatch}
    if lane not in state.lanes or operation not in operations or not isinstance(payload, Mapping):
        raise ValueError('invalid campaign lane, operation or payload')
    if operation == 'begin' and (len(state.dispatches) >= state.max_inflight
                                 or len(state.ledger.records) >= state.max_attempts):
        raise ValueError('campaign concurrency or attempt limit reached')
    current = FeedbackDispatchState(state.lanes[lane], state.ledger, state.dispatches.get(lane))
    updated = operations[operation](current, **dict(payload))
    lanes, dispatches = dict(state.lanes), dict(state.dispatches)
    lanes[lane] = updated.feedback
    if updated.dispatch is None:
        dispatches.pop(lane, None)
    else:
        dispatches[lane] = updated.dispatch
    return FeedbackCampaignState(lanes, dispatches, updated.ledger, state.max_inflight, state.max_attempts)


def restore_feedback_campaign(payload: Mapping[str, Any]) -> FeedbackCampaignState:
    """Restore campaign state; independent journal and source verification are required."""
    if not isinstance(payload, Mapping) or not isinstance(payload.get('lanes'), Mapping) \
            or not isinstance(payload.get('dispatches'), Mapping):
        raise ValueError('invalid saved feedback campaign')
    ledger = restore_resource_ledger(payload['ledger'])
    lanes = {name: restore_feedback(value) for name, value in payload['lanes'].items()}
    dispatches = {name: Dispatch(**value) for name, value in payload['dispatches'].items()}
    return FeedbackCampaignState(lanes, dispatches, ledger, payload['max_inflight'], payload['max_attempts'])
