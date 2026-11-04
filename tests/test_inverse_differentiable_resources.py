"""Check the physical preparation stays within its CPU and memory budget.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import json
from pathlib import Path
import pytest

from phc2d.inverse.differentiable_resources import plan_physical_resources


def test_twelve_candidate_roles_have_bounded_allocation():
    path = Path(__file__).resolve().parents[1]/'configs/research/exp_inv_012_resources_v1.json'
    resources = json.loads(path.read_text())
    manifest = {'selection_lock_sha256': 'lock', 'rows': [{} for _ in range(11)]}
    plan = plan_physical_resources(manifest, resources, manifest_sha256='manifest')
    assert plan['status'] == 'prepared_not_submitted'
    assert [(s['start'], s['end']) for s in plan['shards']] == [(0, 3), (3, 6), (6, 9), (9, 11)]
    assert plan['max_allocated_core_hours'] == 40
    assert plan['workers_per_shard'] <= plan['cpus_per_shard']
    with pytest.raises(ValueError, match='budget'):
        plan_physical_resources({**manifest, 'rows': [{} for _ in range(13)]}, resources,
                                manifest_sha256='manifest')
