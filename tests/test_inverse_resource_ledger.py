"""Tests for explicit inverse design compute budgets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import json
from dataclasses import asdict

import pytest

from phc2d.inverse.resource_ledger import (
    ResourceAmount, ResourceLedger, release_unstarted, reserve_resources, settle_resources,
    settle_slurm_resources, settle_slurm_worker_group, restore_resource_ledger, audit_slurm_ledger,
)


def test_worker_group_accounts_one_allocation_and_keeps_worker_failure():
    ledger = reserve_resources(ResourceLedger(ResourceAmount(40, 0)), task_id='batch', amount=ResourceAmount(40, 0))
    workers = {'slot-a': 'a'*64, 'slot-b': 'b'*64}
    outcomes = {'slot-a': {'candidate_sha256': 'a'*64, 'exit_code': 0, 'result_sha256': 'c'*64},
                'slot-b': {'candidate_sha256': 'b'*64, 'exit_code': 1, 'result_sha256': None}}
    args = {'task_id': 'batch', 'cluster_id': 'm9', 'job_id': '100',
            'accounting_bytes': b'100|FAILED|10|4|cpu=4,node=1\n100.batch|FAILED|10|4|cpu=4,node=1\n',
            'expected_gpus': 0, 'expected_workers': workers, 'worker_outcomes': outcomes}
    completed, audit = settle_slurm_worker_group(ledger, **args)
    assert completed.totals()['actual'] == ResourceAmount(40, 0)
    assert completed.records[0].allocation_ids == ('m9:100',)
    assert audit['worker_count'] == 2 and audit['failed_workers'] == ['slot-b']
    assert audit['allocation_state'] == 'FAILED'
    with pytest.raises(ValueError):
        settle_slurm_worker_group(completed, **args)
    second = reserve_resources(completed, task_id='second-batch', amount=ResourceAmount(0, 0))
    with pytest.raises(ValueError, match='already accounted'):
        settle_slurm_worker_group(second, **dict(args, task_id='second-batch'))
    with pytest.raises(ValueError, match='roster'):
        settle_slurm_worker_group(ledger, **dict(args, worker_outcomes={'slot-a': outcomes['slot-a']}))
    bad = {**outcomes, 'slot-a': {**outcomes['slot-a'], 'candidate_sha256': 'd'*64}}
    with pytest.raises(ValueError, match='identity'):
        settle_slurm_worker_group(ledger, **dict(args, worker_outcomes=bad))
    bad = {**outcomes, 'slot-a': {**outcomes['slot-a'], 'result_sha256': None}}
    with pytest.raises(ValueError, match='result digest'):
        settle_slurm_worker_group(ledger, **dict(args, worker_outcomes=bad))
    with pytest.raises(ValueError, match='roster'):
        settle_slurm_worker_group(ledger, **dict(args, expected_workers={'slot-a': 'a'*64, 'slot-b': 'a'*64}))
    with pytest.raises(ValueError, match='not terminal'):
        settle_slurm_worker_group(ledger, **dict(args, accounting_bytes=b'100|RUNNING|10|4|cpu=4,node=1\n'))
    overrun, _ = settle_slurm_worker_group(ledger, **dict(args,
        accounting_bytes=b'100|FAILED|11|4|cpu=4,node=1\n'))
    assert overrun.totals()['actual'] == ResourceAmount(44, 0)
    assert overrun.status() == 'budget_exceeded'


def test_reservation_and_settlement_account_for_cpu_and_gpu_separately():
    ledger = ResourceLedger(ResourceAmount(100, 20))
    ledger = reserve_resources(ledger, task_id='cpu', amount=ResourceAmount(50, 0))
    ledger = reserve_resources(ledger, task_id='gpu', amount=ResourceAmount(10, 10))
    assert ledger.totals()['reserved'] == ResourceAmount(60, 10)
    ledger = settle_resources(ledger, task_id='cpu', actual=ResourceAmount(30, 0), source='slurm:cpu')
    assert ledger.totals()['actual'] == ResourceAmount(30, 0)
    assert ledger.totals()['reserved'] == ResourceAmount(10, 10)
    with pytest.raises(ValueError, match='unique'):
        reserve_resources(ledger, task_id='cpu', amount=ResourceAmount(1, 0))
    with pytest.raises(ValueError, match='exceeds'):
        reserve_resources(ledger, task_id='too-big', amount=ResourceAmount(61, 0))


def test_actual_overrun_is_recorded_and_blocks_further_reservation():
    ledger = reserve_resources(ResourceLedger(ResourceAmount(10, 0)), task_id='cpu', amount=ResourceAmount(5, 0))
    ledger = settle_resources(ledger, task_id='cpu', actual=ResourceAmount(12, 0), source='slurm:actual')
    assert ledger.status() == 'budget_exceeded'
    assert ledger.totals()['actual'] == ResourceAmount(12, 0)
    with pytest.raises(ValueError, match='exceeds'):
        reserve_resources(ledger, task_id='next', amount=ResourceAmount(0, 0))


def test_release_requires_external_nonexecution_confirmation():
    ledger = reserve_resources(ResourceLedger(ResourceAmount(10, 0)), task_id='cpu', amount=ResourceAmount(5, 0))
    with pytest.raises(ValueError, match='confirmed'):
        release_unstarted(ledger, task_id='cpu', confirmed_not_started=False, confirmation_source='slurm')
    released = release_unstarted(ledger, task_id='cpu', confirmed_not_started=True, confirmation_source='slurm')
    assert released.records[0].source == 'slurm'
    assert released.totals()['reserved'] == ResourceAmount(0, 0)
    with pytest.raises(ValueError, match='unique'):
        reserve_resources(released, task_id='cpu', amount=ResourceAmount(1, 0))


@pytest.mark.parametrize('amount', [(-1, 0), (0, float('nan')), (float('inf'), 0)])
def test_invalid_usage_is_rejected(amount):
    with pytest.raises(ValueError, match='finite'):
        ResourceAmount(*amount)


def test_slurm_settlement_counts_failures_and_rejects_duplicate_allocations():
    raw = (b'10_0|COMPLETED|10|3|cpu=3,node=1\n'
           b'10_0.batch|COMPLETED|10|3|cpu=3,node=1\n'
           b'10_1|FAILED|5|3|cpu=3,node=1\n')
    ledger = ResourceLedger(ResourceAmount(1000, 100))
    ledger = reserve_resources(ledger, task_id='first', amount=ResourceAmount(100, 0))
    ledger = reserve_resources(ledger, task_id='second', amount=ResourceAmount(100, 0))
    ledger = settle_slurm_resources(ledger, task_id='first', cluster_id='m9', accounting_bytes=raw,
                                    expected_gpus={'10_0': 0, '10_1': 0})
    assert ledger.totals()['actual'] == ResourceAmount(45, 0)
    assert ledger.records[0].allocation_ids == ('m9:10_0', 'm9:10_1')
    assert ledger.records[0].source.startswith('slurm:m9:sha256:')
    with pytest.raises(ValueError, match='already accounted'):
        settle_slurm_resources(ledger, task_id='second', cluster_id='m9', accounting_bytes=raw,
                               expected_gpus={'10_0': 0, '10_1': 0})
    # IDs from different clusters are distinct allocations.
    other = settle_slurm_resources(ledger, task_id='second', cluster_id='n56', accounting_bytes=raw,
                                   expected_gpus={'10_0': 0, '10_1': 0})
    assert other.totals()['actual'] == ResourceAmount(90, 0)


def test_slurm_running_job_cannot_settle_reserved_budget():
    ledger = reserve_resources(ResourceLedger(ResourceAmount(100, 10)), task_id='job',
                               amount=ResourceAmount(30, 5))
    with pytest.raises(ValueError, match='not terminal'):
        settle_slurm_resources(ledger, task_id='job', cluster_id='n56',
            accounting_bytes=b'100|RUNNING|1|6|cpu=6,gres/gpu=1\n', expected_gpus={'100': 1})
    assert ledger.records[0].state == 'reserved'


def test_audit_replays_raw_accounting_and_rejects_tampered_or_open_ledger():
    ledger = reserve_resources(ResourceLedger(ResourceAmount(100, 10)), task_id='job', amount=ResourceAmount(50, 5))
    raw = b'10|FAILED|4|3|cpu=3,gres/gpu=1\n'
    evidence = {'job': {'cluster_id': 'n56', 'raw': raw, 'expected_gpus': {'10': 1}}}
    with pytest.raises(ValueError, match='pending or released'):
        audit_slurm_ledger(ledger, accounting_by_task={})
    ledger = settle_slurm_resources(ledger, task_id='job', cluster_id='n56',
                                     accounting_bytes=raw, expected_gpus={'10': 1})
    result = audit_slurm_ledger(ledger, accounting_by_task=evidence, require_settled=True)
    assert result['cpu_core_seconds'] == 12 and result['gpu_seconds'] == 4
    assert result['allocation_ids'] == ('n56:10',)
    with pytest.raises(ValueError, match='does not match raw'):
        audit_slurm_ledger(ledger, accounting_by_task={'job': dict(evidence['job'], raw=raw.replace(b'|4|', b'|5|'))})
    with pytest.raises(ValueError, match='does not match settled'):
        audit_slurm_ledger(ledger, accounting_by_task={})
    with pytest.raises(ValueError, match='no settled accounting'):
        audit_slurm_ledger(ResourceLedger(ResourceAmount(100, 10)), accounting_by_task={}, require_settled=True)
    over = reserve_resources(ResourceLedger(ResourceAmount(10, 10)), task_id='job', amount=ResourceAmount(5, 5))
    over = settle_slurm_resources(over, task_id='job', cluster_id='n56', accounting_bytes=raw,
                                  expected_gpus={'10': 1})
    with pytest.raises(ValueError, match='exceeds its budget'):
        audit_slurm_ledger(over, accounting_by_task=evidence)


def test_ledger_round_trip_preserves_overrun_reservations_and_accounted_jobs():
    ledger = ResourceLedger(ResourceAmount(100, 10))
    ledger = reserve_resources(ledger, task_id='done', amount=ResourceAmount(20, 0))
    ledger = reserve_resources(ledger, task_id='pending', amount=ResourceAmount(10, 5))
    raw = b'10|FAILED|40|3|cpu=3,node=1\n'
    ledger = settle_slurm_resources(ledger, task_id='done', cluster_id='m9',
                                    accounting_bytes=raw, expected_gpus={'10': 0})
    assert ledger.status() == 'budget_exceeded'
    restored = restore_resource_ledger(json.loads(json.dumps(asdict(ledger))))
    assert restored == ledger
    assert restored.totals()['reserved'] == ResourceAmount(10, 5)
    with pytest.raises(ValueError, match='already accounted'):
        settle_slurm_resources(restored, task_id='pending', cluster_id='m9',
                               accounting_bytes=raw, expected_gpus={'10': 0})


def test_ledger_restore_rejects_corrupt_states_and_duplicate_allocation():
    ledger = ResourceLedger(ResourceAmount(100, 0))
    ledger = reserve_resources(ledger, task_id='a', amount=ResourceAmount(10, 0))
    ledger = settle_slurm_resources(ledger, task_id='a', cluster_id='m9',
        accounting_bytes=b'10|COMPLETED|1|3|cpu=3\n', expected_gpus={'10': 0})
    data = json.loads(json.dumps(asdict(ledger)))
    duplicate = dict(data['records'][0], task_id='b')
    data['records'].append(duplicate)
    with pytest.raises(ValueError, match='already accounted'):
        restore_resource_ledger(data)
    data['records'].pop()
    data['records'][0]['actual']['cpu_core_seconds'] = True
    with pytest.raises(ValueError, match='finite'):
        restore_resource_ledger(data)
