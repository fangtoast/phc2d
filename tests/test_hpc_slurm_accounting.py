"""Verify scheduler allocation costs and avoid job-step double counting.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib

import pytest

from phc2d.hpc.slurm_accounting import parse_terminal_allocations


def test_array_and_gpu_allocations_ignore_steps_and_keep_failures():
    raw = (b'100_0|COMPLETED|120|3|billing=3,cpu=3,mem=8G,node=1\n'
           b'100_0.batch|COMPLETED|120|3|cpu=3,mem=8G,node=1\n'
           b'100_0.extern|COMPLETED|120|3|cpu=3,mem=8G,node=1\n'
           b'100_1|FAILED|60|3|billing=3,cpu=3,mem=8G,node=1\n'
           b'200|COMPLETED|30|6|billing=6,cpu=6,gres/gpu=1,mem=60G,node=1\n')
    result = parse_terminal_allocations(raw, expected_gpus={'100_0': 0, '100_1': 0, '200': 1})
    assert result['source_sha256'] == hashlib.sha256(raw).hexdigest()
    assert result['cpu_core_seconds'] == 120 * 3 + 60 * 3 + 30 * 6
    assert result['gpu_seconds'] == 30
    assert [row.state for row in result['allocations']] == ['COMPLETED', 'FAILED', 'COMPLETED']


def test_typed_gpu_counts_are_not_added_twice():
    raw = b'200|COMPLETED|10|6|cpu=6,gres/gpu=2,gres/gpu:4090=2\n'
    result = parse_terminal_allocations(raw, expected_gpus={'200': 2})
    assert result['gpu_seconds'] == 20
    with pytest.raises(ValueError, match='disagree'):
        parse_terminal_allocations(raw.replace(b'4090=2', b'4090=1'), expected_gpus={'200': 2})


@pytest.mark.parametrize('raw,expected,reason', [
    (b'100|RUNNING|10|3|cpu=3\n', {'100': 0}, 'not terminal'),
    (b'100|COMPLETED|10|3|cpu=3\n', {'100': 1}, 'GPU allocation'),
    (b'100|COMPLETED|10|3|cpu=4\n', {'100': 0}, 'CPU TRES'),
    (b'100|COMPLETED|10|3|cpu=3\n100|COMPLETED|10|3|cpu=3\n', {'100': 0}, 'duplicate'),
    (b'100|COMPLETED|10|3|cpu=3\n', {'100': 0, '101': 0}, 'missing'),
    (b'100|COMPLETED|10|3|cpu=3\n101|COMPLETED|10|3|cpu=3\n', {'100': 0}, 'unrequested'),
    (b'100|COMPLETED|Unknown|3|cpu=3\n', {'100': 0}, 'unavailable'),
])
def test_unsettled_or_inconsistent_records_fail_closed(raw, expected, reason):
    with pytest.raises(ValueError, match=reason):
        parse_terminal_allocations(raw, expected_gpus=expected)
