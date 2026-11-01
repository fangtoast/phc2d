"""Test bounded execution and evidence handling for packed inverse cases.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json
import threading
import time
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from phc2d.inverse.packed_refine import run_packed_refinement


def arguments(tmp_path):
    manifest = tmp_path / 'manifest.json'
    protocol = tmp_path / 'protocol.json'
    manifest.write_text(json.dumps({'rows': [{}, {}, {}]}))
    protocol.write_text('{}')
    return dict(manifest=manifest, manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                protocol=protocol, protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                output_root=tmp_path/'output', start=0, end=3, workers=2, cpu_limit=2,
                memory_budget_mib=2000, worker_memory_mib=500, setup_reserve_mib=1000)


def test_packed_cases_run_concurrently_and_record_failure(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    lock, active, peak, seen = threading.Lock(), 0, 0, []

    def fake_run(command, *, stdout, stderr, env, check):
        nonlocal active, peak
        index = int(command[command.index('--case-index') + 1])
        assert all(env[k] == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
        with lock:
            active += 1
            peak = max(peak, active)
            seen.append(index)
        time.sleep(.05)
        with lock:
            active -= 1
        return SimpleNamespace(returncode=1 if index == 1 else 0)

    monkeypatch.setattr('phc2d.inverse.packed_cases.subprocess.run', fake_run)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    monkeypatch.delenv('SLURM_MEM_PER_NODE', raising=False)
    result = run_packed_refinement(**args)
    assert peak == 2 and sorted(seen) == [0, 1, 2]
    assert not result['all_succeeded']
    assert [entry['returncode'] for entry in result['cases']] == [0, 1, 0]
    assert json.loads((args['output_root']/'packed_0000_0003'/'summary.json').read_text()) == result
    with pytest.raises(FileExistsError):
        run_packed_refinement(**args)


def test_packed_rejects_overcommit_scheduler_mismatch_and_changed_input(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    with pytest.raises(ValueError, match='budget'):
        run_packed_refinement(**dict(args, workers=3))
    with pytest.raises(ValueError, match='budget'):
        run_packed_refinement(**dict(args, memory_budget_mib=1999))
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '1')
    with pytest.raises(ValueError, match='CPU limit'):
        run_packed_refinement(**args)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '2')
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '1999')
    with pytest.raises(ValueError, match='memory budget'):
        run_packed_refinement(**args)
    monkeypatch.delenv('SLURM_MEM_PER_NODE')
    args['protocol'].write_text('{"changed": true}')
    with pytest.raises(ValueError, match='hash mismatch'):
        run_packed_refinement(**args)


def test_packed_refuses_existing_physical_output_before_launch(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    monkeypatch.delenv('SLURM_MEM_PER_NODE', raising=False)
    (args['output_root']/'case_0001').mkdir(parents=True)
    with pytest.raises(FileExistsError, match='case output exists'):
        run_packed_refinement(**args)
    assert not (args['output_root']/'packed_0000_0003').exists()


@pytest.mark.skipif(not os.access('/usr/bin/time', os.X_OK), reason='GNU time required')
def test_optional_worker_resource_reporting_measures_each_real_subprocess(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    actual_run = subprocess.run
    def substitute_solver(command, *, stdout, stderr, env, check):
        index = int(command[command.index('--case-index') + 1])
        program = f"data=bytearray(1000000); sum(data); raise SystemExit({int(index == 1)})"
        return actual_run([*command[:6], sys.executable, '-c', program],
                          stdout=stdout, stderr=stderr, env=env, check=check)
    monkeypatch.setattr('phc2d.inverse.packed_cases.subprocess.run', substitute_solver)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    monkeypatch.delenv('SLURM_MEM_PER_NODE', raising=False)
    result = run_packed_refinement(**args, worker_resource_report=True)
    assert not result['all_succeeded'] and result['all_worker_resources_reported']
    assert [row['returncode'] for row in result['cases']] == [0, 1, 0]
    assert all(row['resource_status']=='reported' and row['max_rss_kib'] > 0
               and row['elapsed_seconds'] >= 0 for row in result['cases'])
    assert len(list((args['output_root']/'packed_0000_0003').glob('*.resources'))) == 3


@pytest.mark.skipif(not os.access('/usr/bin/time', os.X_OK), reason='GNU time required')
def test_invalid_worker_resource_output_is_visible(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    def fake_run(command, *, stdout, stderr, env, check):
        path = command[command.index('-o')+1]
        with open(path, 'w') as stream:
            stream.write('invalid')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('phc2d.inverse.packed_cases.subprocess.run', fake_run)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    monkeypatch.delenv('SLURM_MEM_PER_NODE', raising=False)
    result = run_packed_refinement(**args, worker_resource_report=True)
    assert result['all_succeeded'] and not result['all_worker_resources_reported']
    assert all(row['resource_status']=='unavailable_or_invalid' for row in result['cases'])
