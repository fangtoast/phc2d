"""Reconcile terminal search tasks and freeze paired choices from their traces.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from phc2d.inverse.paired_search_materialization import audit_paired_search_requests
from phc2d.inverse.trace_selection import lock_paired_trace_selection


def collect_paired_search_locks(
    batch: Mapping[str, Any], *, request_root: Path, output_root: Path,
    job_records: Sequence[Mapping[str, Any]], expected_k_points_sha256: str, random_choice_seed: int,
) -> dict[str, Any]:
    """Recompute every trace lock after the caller authenticates scheduler records.

    Running or unknown jobs remain incomplete and are never restarted here.
    Physical outcomes and lock chronology must be checked independently.
    """
    materialized = audit_paired_search_requests(batch,request_root)
    requests = batch['requests']
    if (not isinstance(expected_k_points_sha256, str) or len(expected_k_points_sha256) != 64
            or any(ch not in '0123456789abcdef' for ch in expected_k_points_sha256)
            or isinstance(random_choice_seed, bool) or not isinstance(random_choice_seed, int) or random_choice_seed < 0):
        raise ValueError('invalid frozen search collection contract')
    by_index: dict[int, Mapping[str, Any]] = {}
    for record in job_records:
        index = record.get('request_index')
        if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(requests)
                or index in by_index or not isinstance(record.get('job_id'), str) or not record['job_id']):
            raise ValueError('invalid or duplicate scheduler record')
        by_index[index] = record
    if set(by_index) != set(range(len(requests))):
        raise ValueError('scheduler record roster incomplete')
    statuses, locks = [], []
    for index, request in enumerate(requests):
        job = by_index[index]
        state, exit_code = job.get('state'), job.get('exit_code')
        if state != 'COMPLETED' or exit_code != '0:0':
            terminal_failure = state in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
            statuses.append({'request_index':index,'job_id':job['job_id'],
                             'status':'terminal_failure' if terminal_failure or state == 'COMPLETED' else 'not_terminal',
                             'state':state,'exit_code':exit_code})
            continue
        sid, model_seed, search_seed = request['source_index'], request['model_seed'], request['search_seed']
        root = Path(request_root)
        config_raw = (root/f'configs/source_{sid:04d}_model_{model_seed}.json').read_bytes()
        source_raw = (root/f'sources/source_{sid:04d}.json').read_bytes()
        try:
            lock = lock_paired_trace_selection(Path(output_root)/f'task_{index}',config_bytes=config_raw,
                source_bytes=source_raw,checkpoint_sha256=request['config']['checkpoint_sha256'],
                source_index=sid,model_seed=model_seed,search_seed=search_seed,random_choice_seed=random_choice_seed)
            if lock['checkpoint_k_points_sha256'] != expected_k_points_sha256:
                raise ValueError('checkpoint k-point identity mismatch')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            statuses.append({'request_index':index,'job_id':job['job_id'],
                             'status':'invalid_artifact','reason':str(exc)})
            continue
        locks.append(lock)
        statuses.append({'request_index':index,'job_id':job['job_id'],'status':lock['status']})
    if any(item['status'] == 'not_terminal' for item in statuses):
        overall = 'incomplete_running_or_unknown'
    elif any(item['status'] in ('terminal_failure','invalid_artifact') for item in statuses):
        overall = 'unscoreable_failed_tasks_or_artifacts'
    elif any(item['status'] != 'locked_surrogate_only' for item in statuses):
        overall = 'unscoreable_search_traces'
    else:
        overall = 'locked_surrogate_only'
    return {'status':overall, 'physical_validation':'not_run', 'request_manifest_sha256':materialized['manifest_sha256'],
            'request_count':len(requests), 'statuses':statuses, 'locks':locks}
