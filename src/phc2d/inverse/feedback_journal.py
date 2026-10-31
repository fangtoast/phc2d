"""Persist and reconcile a single asynchronous robust feedback attempt.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.hpc.slurm_accounting import parse_terminal_allocations
from phc2d.inverse.feedback_handoff import consume_robust_solver_result
from phc2d.inverse.feedback_state import FeedbackState, finish_feedback, restore_feedback


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _encoded(raw: bytes) -> str:
    return base64.b64encode(raw).decode('ascii')


def _decode(value: str, expected: str) -> bytes:
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid journal encoding') from exc
    if _hash(raw) != expected:
        raise ValueError('journal input digest mismatch')
    return raw


def _serialize(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def _write_once(path: Path, raw: bytes) -> None:
    """Never overwrite a recorded request or receipt; partial writes fail closed."""
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_request(request: dict[str, Any]) -> tuple[FeedbackState, bytes, bytes]:
    if request.get('schema') != 'robust_feedback_round_v1':
        raise ValueError('unknown journal schema')
    manifest_bytes = _decode(request['manifest'], request['manifest_sha256'])
    protocol_bytes = _decode(request['protocol'], request['protocol_sha256'])
    manifest, protocol = json.loads(manifest_bytes), json.loads(protocol_bytes)
    state = restore_feedback(request['pending_state'])
    if (state.pending is None or manifest.get('status') != 'feedback_robust_solver_request'
            or len(manifest.get('rows', [])) != 1 or not isinstance(protocol, dict)):
        raise ValueError('journal needs one pending robust candidate')
    row = manifest['rows'][0]
    structure = FullCellStructureSpec.from_mapping(row['structure'])
    if (row.get('label') != 'candidate' or structure.parameter_hash != row['parameter_hash']
            or structure.parameter_hash != state.pending.parameter_hash
            or manifest.get('surrogate_contract_sha256') != state.pending.surrogate_contract_sha256
            or manifest.get('target_frequency') != state.target_frequency
            or manifest.get('lower_band') != state.incumbent_observation['row']['lower_band_one_based']):
        raise ValueError('journal candidate identity or target mismatch')
    return state, manifest_bytes, protocol_bytes


def create_robust_round(directory: str | Path, *, pending: FeedbackState,
                        manifest_bytes: bytes, protocol_bytes: bytes) -> Path:
    """Persist a pending request before dispatch. The directory must be new.

    A controller must use a single writer and a distinct directory per attempt.
    An interrupted or incomplete request is an error requiring reconciliation,
    never permission to submit a duplicate solver job.
    """
    request = {'schema': 'robust_feedback_round_v1', 'pending_state': asdict(pending),
               'manifest': _encoded(manifest_bytes), 'manifest_sha256': _hash(manifest_bytes),
               'protocol': _encoded(protocol_bytes), 'protocol_sha256': _hash(protocol_bytes)}
    _validate_request(request)
    target = Path(directory)
    target.mkdir(parents=False, exist_ok=False)
    _write_once(target / 'request.json', _serialize(request))
    return target


def _load_request(directory: Path) -> tuple[dict[str, Any], bytes]:
    raw = (directory / 'request.json').read_bytes()
    request = json.loads(raw)
    _validate_request(request)
    return request, raw


def reconcile_robust_round(directory: str | Path, *, solver_result_bytes: bytes) -> FeedbackState:
    """Verify a solver result and persist its decision exactly once.

    Replaying the identical result is idempotent. A different result for the
    same request is rejected, even if it would imply the same decision.
    """
    target = Path(directory)
    request, request_bytes = _load_request(target)
    state, manifest_bytes, protocol_bytes = _validate_request(request)
    finished = consume_robust_solver_result(state, manifest_bytes=manifest_bytes,
        expected_manifest_sha256=request['manifest_sha256'], protocol_bytes=protocol_bytes,
        protocol_sha256=request['protocol_sha256'], result=json.loads(solver_result_bytes))
    receipt = {'schema': 'robust_feedback_receipt_v1', 'request_sha256': _hash(request_bytes),
               'solver_result': _encoded(solver_result_bytes), 'solver_result_sha256': _hash(solver_result_bytes),
               'completed_state': asdict(finished)}
    raw = _serialize(receipt)
    path = target / 'receipt.json'
    try:
        _write_once(path, raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError('conflicting or incomplete solver receipt') from None
    return finished


def _failed_attempt(pending: FeedbackState, accounting: bytes, cluster: str,
                    job: str, gpus: int) -> FeedbackState:
    if not isinstance(cluster, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', cluster):
        raise ValueError('invalid scheduler cluster')
    allocation = parse_terminal_allocations(accounting, expected_gpus={job: gpus})['allocations'][0]
    if allocation.state == 'COMPLETED':
        raise ValueError('completed allocation is not a confirmed scheduler failure')
    return finish_feedback(pending, failure_reason=f'slurm:{cluster}:{job}:{allocation.state}')


def reconcile_failed_robust_round(directory: str | Path, *, accounting_bytes: bytes,
                                  cluster: str, job: str, allocated_gpus: int) -> FeedbackState:
    """Record one terminal scheduler failure; caller verifies ownership and no requeue.

    The exact scheduler allocation and raw accounting are retained for replay.
    This neither dispatches jobs nor proves that no valid solver output exists.
    """
    target = Path(directory)
    request, request_bytes = _load_request(target)
    pending, _, _ = _validate_request(request)
    finished = _failed_attempt(pending, accounting_bytes, cluster, job, allocated_gpus)
    receipt = {'schema': 'robust_feedback_failure_v1', 'request_sha256': _hash(request_bytes),
               'accounting': _encoded(accounting_bytes), 'accounting_sha256': _hash(accounting_bytes),
               'cluster': cluster, 'job': job, 'allocated_gpus': allocated_gpus,
               'completed_state': asdict(finished)}
    raw = _serialize(receipt)
    path = target / 'receipt.json'
    try:
        _write_once(path, raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError('conflicting or incomplete solver receipt') from None
    return finished


def read_robust_round(directory: str | Path) -> tuple[FeedbackState, str]:
    """Restore a request or recompute a completed decision from its raw result."""
    target = Path(directory)
    request, request_bytes = _load_request(target)
    pending, manifest_bytes, protocol_bytes = _validate_request(request)
    path = target / 'receipt.json'
    if not path.exists():
        return pending, 'pending_reconciliation'
    receipt = json.loads(path.read_bytes())
    if receipt.get('request_sha256') != _hash(request_bytes):
        raise ValueError('receipt request identity mismatch')
    if receipt.get('schema') == 'robust_feedback_receipt_v1':
        result_bytes = _decode(receipt['solver_result'], receipt['solver_result_sha256'])
        finished = consume_robust_solver_result(pending, manifest_bytes=manifest_bytes,
            expected_manifest_sha256=request['manifest_sha256'], protocol_bytes=protocol_bytes,
            protocol_sha256=request['protocol_sha256'], result=json.loads(result_bytes))
    elif receipt.get('schema') == 'robust_feedback_failure_v1':
        accounting = _decode(receipt['accounting'], receipt['accounting_sha256'])
        finished = _failed_attempt(pending, accounting, receipt['cluster'], receipt['job'], receipt['allocated_gpus'])
    else:
        raise ValueError('unknown receipt schema')
    if _serialize(asdict(finished)) != _serialize(receipt['completed_state']):
        raise ValueError('receipt decision does not match solver evidence')
    return finished, 'completed'


def verify_dispatch_round(dispatch_state: Any, directory: str | Path) -> str:
    """Compare a persisted dispatch with a replayed journal without sending jobs.

    ``receipt_ahead`` means a receipt was saved before the dispatch snapshot;
    the caller must reconcile the existing receipt, not repeat the attempt.
    The caller remains responsible for scheduler identity, single-writer
    ownership, validating the initial incumbent, and durable state storage.
    """
    from phc2d.inverse.feedback_dispatch import FeedbackDispatchState

    if not isinstance(dispatch_state, FeedbackDispatchState) or dispatch_state.dispatch is None:
        raise ValueError('active dispatch required')
    item = dispatch_state.dispatch
    target = Path(directory)
    request, _ = _load_request(target)
    pending, _, _ = _validate_request(request)
    if pending.pending.parameter_hash != item.candidate_hash:
        raise ValueError('dispatch and journal candidate differ')
    replayed, status = read_robust_round(target)
    if item.phase != 'awaiting_accounting':
        if pending != dispatch_state.feedback:
            raise ValueError('dispatch and journal pending states differ')
        if status == 'completed':
            if item.phase != 'submitted':
                raise ValueError('receipt exists without a confirmed job')
            receipt = json.loads((target / 'receipt.json').read_bytes())
            if (receipt.get('schema') == 'robust_feedback_failure_v1'
                    and (receipt.get('cluster') != item.cluster_id or receipt.get('job') != item.job_id)):
                raise ValueError('dispatch and scheduler failure receipt differ')
            return 'receipt_ahead'
        return 'in_sync'
    if status != 'completed' or replayed != dispatch_state.feedback:
        raise ValueError('dispatch and journal completed states differ')
    receipt = json.loads((target / 'receipt.json').read_bytes())
    if item.result_sha256 is not None:
        if receipt.get('schema') != 'robust_feedback_receipt_v1' or receipt.get('solver_result_sha256') != item.result_sha256:
            raise ValueError('dispatch and solver receipt differ')
    elif (receipt.get('schema') != 'robust_feedback_failure_v1'
          or receipt.get('cluster') != item.cluster_id or receipt.get('job') != item.job_id
          or replayed.history[-1]['reason'] != f'slurm:{item.cluster_id}:{item.job_id}:{item.terminal_failure}'):
        raise ValueError('dispatch and scheduler failure receipt differ')
    return 'in_sync'


def verify_robust_journal_chain(initial: FeedbackState,
                                directories: list[str | Path]) -> tuple[FeedbackState, str]:
    """Replay consecutive rounds from one caller-validated initial incumbent.

    Every complete round is recomputed from its raw solver result. Its pending
    state must extend the previous decision exactly; at most the final round
    may remain pending. The caller separately validates the initial incumbent
    and single-writer ownership before any external job dispatch.
    """
    current = restore_feedback(asdict(initial))
    if current.pending is not None or current.history or current.remaining_evaluations != current.total_candidate_budget:
        raise ValueError('chain needs an unspent initial state')
    paths = [Path(path).resolve() for path in directories]
    if len(paths) != len(set(paths)):
        raise ValueError('a journal cannot appear twice in one chain')
    for index, path in enumerate(paths):
        request, _ = _load_request(path)
        pending, _, _ = _validate_request(request)
        if replace(pending, pending=None) != current:
            raise ValueError('journal does not extend the previous verified state')
        result, status = read_robust_round(path)
        if status == 'pending_reconciliation':
            if index != len(paths) - 1:
                raise ValueError('an unfinished round must be last')
            return result, status
        current = result
    return current, 'budget_exhausted' if current.remaining_evaluations == 0 else 'ready'


def _apply_dispatch_event(state: Any, operation: str, payload: dict[str, Any]) -> Any:
    """Replay one pure dispatch transition; no scheduler action is performed."""
    from phc2d.inverse import feedback_dispatch as dispatch
    from phc2d.inverse.resource_ledger import ResourceAmount

    if not isinstance(payload, dict):
        raise ValueError('dispatch event payload must be an object')
    fields = {
        'begin': {'proposal', 'task_id', 'cluster_id', 'reservation'},
        'submission_unknown': {'reason'},
        'confirm': {'job_id', 'confirmation_source'},
        'release': {'confirmed_not_started', 'confirmed_no_future_execution', 'confirmation_source'},
        'result': {'manifest', 'manifest_sha256', 'protocol', 'protocol_sha256', 'result'},
        'failure': {'failure'},
        'settle': {'accounting', 'allocated_gpus'},
    }
    if not isinstance(operation, str) or operation not in fields or set(payload) != fields[operation]:
        raise ValueError('unknown dispatch operation or unexpected payload fields')
    if operation == 'begin':
        reservation = payload['reservation']
        if not isinstance(reservation, dict) or set(reservation) != {'cpu_core_seconds', 'gpu_seconds'}:
            raise ValueError('invalid dispatch reservation')
        return dispatch.begin_dispatch(state, proposal=payload['proposal'], task_id=payload['task_id'],
                                       cluster_id=payload['cluster_id'], reservation=ResourceAmount(**reservation))
    if operation == 'submission_unknown':
        return dispatch.mark_submission_unknown(state, reason=payload['reason'])
    if operation == 'confirm':
        return dispatch.confirm_submission(state, **payload)
    if operation == 'release':
        return dispatch.resolve_unstarted(state, **payload)
    if operation == 'failure':
        return dispatch.record_terminal_failure(state, **payload)
    try:
        if operation == 'result':
            return dispatch.record_robust_result(state, manifest_bytes=base64.b64decode(payload['manifest'], validate=True),
                manifest_sha256=payload['manifest_sha256'], protocol_bytes=base64.b64decode(payload['protocol'], validate=True),
                protocol_sha256=payload['protocol_sha256'], result_bytes=base64.b64decode(payload['result'], validate=True))
        return dispatch.settle_dispatch(state, accounting_bytes=base64.b64decode(payload['accounting'], validate=True),
                                        allocated_gpus=payload['allocated_gpus'])
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid dispatch event or encoded evidence') from exc


def create_dispatch_journal(directory: str | Path, *, initial: Any) -> str:
    """Create a new journal and return its initial hash for external custody.

    The caller validates the initial incumbent, scheduler evidence and single
    writer ownership; this journal neither submits jobs nor authenticates the
    external origin of subsequently supplied evidence.
    """
    from phc2d.inverse.feedback_dispatch import FeedbackDispatchState, restore_feedback_dispatch

    if not isinstance(initial, FeedbackDispatchState) or initial.dispatch is not None:
        raise ValueError('initial dispatch state must not contain an active attempt')
    restored = restore_feedback_dispatch(asdict(initial))
    raw = _serialize({'schema': 'feedback_dispatch_journal_v1', 'state': asdict(restored)})
    target = Path(directory)
    target.mkdir(parents=False, exist_ok=False)
    (target / 'events').mkdir()
    _write_once(target / 'initial.json', raw)
    return _hash(raw)


def replay_dispatch_journal(directory: str | Path, *, initial_sha256: str) -> tuple[Any, str, int]:
    """Recompute all transitions and return state, current hash and event count.

    A trusted externally retained head is needed to detect a rewritten or
    truncated otherwise-valid journal. Incomplete writes fail closed.
    """
    from phc2d.inverse.feedback_dispatch import restore_feedback_dispatch

    target = Path(directory)
    raw = (target / 'initial.json').read_bytes()
    if _hash(raw) != initial_sha256:
        raise ValueError('dispatch journal initial hash differs')
    initial = json.loads(raw)
    if set(initial) != {'schema', 'state'} or initial['schema'] != 'feedback_dispatch_journal_v1':
        raise ValueError('invalid dispatch journal initial record')
    state = restore_feedback_dispatch(initial['state'])
    if state.dispatch is not None:
        raise ValueError('dispatch journal begins with an active attempt')
    entries = sorted((target / 'events').iterdir(), key=lambda path: path.name)
    head = initial_sha256
    for number, path in enumerate(entries, start=1):
        if path.name != f'{number:08d}.json' or not path.is_file() or path.is_symlink():
            raise ValueError('dispatch journal event sequence differs')
        event_raw = path.read_bytes()
        event = json.loads(event_raw)
        if (set(event) != {'schema', 'number', 'previous_sha256', 'operation', 'payload', 'state_sha256'}
                or event['schema'] != 'feedback_dispatch_event_v1' or isinstance(event['number'], bool)
                or event['number'] != number
                or event['previous_sha256'] != head):
            raise ValueError('dispatch journal event identity differs')
        state = _apply_dispatch_event(state, event['operation'], event['payload'])
        if _hash(_serialize(asdict(state))) != event['state_sha256']:
            raise ValueError('dispatch journal transition differs')
        head = _hash(event_raw)
    return state, head, len(entries)


def append_dispatch_event(directory: str | Path, *, initial_sha256: str, expected_head: str,
                          operation: str, payload: dict[str, Any]) -> tuple[Any, str]:
    """Validate and append one immutable transition under single-writer control."""
    state, head, count = replay_dispatch_journal(directory, initial_sha256=initial_sha256)
    if head != expected_head:
        raise ValueError('dispatch journal head differs; reconcile before writing')
    next_state = _apply_dispatch_event(state, operation, payload)
    event = {'schema': 'feedback_dispatch_event_v1', 'number': count + 1, 'previous_sha256': head,
             'operation': operation, 'payload': payload, 'state_sha256': _hash(_serialize(asdict(next_state)))}
    raw = _serialize(event)
    _write_once(Path(directory) / 'events' / f'{count + 1:08d}.json', raw)
    return next_state, _hash(raw)


def summarize_dispatch_journal(directory: str | Path, *, initial_sha256: str,
                               expected_head: str) -> dict[str, Any]:
    """Inspect a chain against externally retained initial and head hashes."""
    state, head, count = replay_dispatch_journal(directory, initial_sha256=initial_sha256)
    if head != expected_head:
        raise ValueError('dispatch journal head differs; reconcile before use')
    totals = state.ledger.totals()
    item = state.dispatch
    return {'status': 'chain_verified', 'event_count': count, 'head_sha256': head,
            'phase': item.phase if item else 'idle', 'task_id': item.task_id if item else None,
            'cluster_id': item.cluster_id if item else None, 'job_id': item.job_id if item else None,
            'incumbent_hash': state.feedback.incumbent_hash,
            'pending_hash': state.feedback.pending.parameter_hash if state.feedback.pending else None,
            'remaining_evaluations': state.feedback.remaining_evaluations,
            'budget_status': state.ledger.status(), 'limits': asdict(state.ledger.limits),
            'actual': asdict(totals['actual']), 'reserved': asdict(totals['reserved']),
            'evidence_scope': 'journal_replay_only_initial_and_scheduler_evidence_not_independently_verified'}


def _status_summary(state: FeedbackState, status: str, scope: str) -> dict[str, Any]:
    return {'status': status, 'incumbent_hash': state.incumbent_hash,
            'remaining_evaluations': state.remaining_evaluations,
            'pending_hash': state.pending.parameter_hash if state.pending else None,
            'last_decision': state.history[-1] if state.history else None,
            'evidence_scope': scope}


def summarize_robust_round(directory: str | Path) -> dict[str, Any]:
    """Read and recompute one round without certifying its incoming incumbent."""
    state, status = read_robust_round(directory)
    return _status_summary(state, status, 'single_round_replay_initial_incumbent_not_independently_verified')


def summarize_robust_chain(initial: FeedbackState, directories: list[str | Path]) -> dict[str, Any]:
    """Replay an ordered chain; caller separately verifies the initial anchor."""
    state, status = verify_robust_journal_chain(initial, directories)
    return _status_summary(state, status, 'ordered_chain_replay_initial_incumbent_not_independently_verified')


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect or reconcile one robust feedback journal')
    subparsers = parser.add_subparsers(dest='action', required=True)
    inspect = subparsers.add_parser('inspect', help='Verify saved evidence and show the current state')
    inspect.add_argument('--directory', required=True)
    reconcile = subparsers.add_parser('reconcile', help='Verify one completed solver result')
    reconcile.add_argument('--directory', required=True)
    reconcile.add_argument('--result', required=True)
    chain = subparsers.add_parser('inspect-chain', help='Verify an initial state and ordered journal chain')
    chain.add_argument('--initial-state', required=True)
    chain.add_argument('--round', action='append', default=[])
    args = parser.parse_args()
    if args.action == 'inspect':
        state, status = read_robust_round(args.directory)
    elif args.action == 'reconcile':
        state = reconcile_robust_round(args.directory, solver_result_bytes=Path(args.result).read_bytes())
        status = 'completed'
    else:
        initial = restore_feedback(json.loads(Path(args.initial_state).read_bytes()))
        state, status = verify_robust_journal_chain(initial, args.round)
    print(json.dumps({'status': status, 'incumbent_hash': state.incumbent_hash,
                      'remaining_evaluations': state.remaining_evaluations,
                      'pending_hash': state.pending.parameter_hash if state.pending else None,
                      'last_decision': state.history[-1] if state.history else None},
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
