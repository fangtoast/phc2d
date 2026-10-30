"""Tests for explicit inverse design compute budgets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

from phc2d.inverse.resource_ledger import (
    ResourceAmount, ResourceLedger, release_unstarted, reserve_resources, settle_resources,
)


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
