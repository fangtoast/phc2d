"""Check the physical preparation stays within its CPU and memory budget.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
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


def test_three_source_budget_caps_array_concurrency():
    root = Path(__file__).resolve().parents[1]
    resources = json.loads((root/'configs/research/exp_inv_013_resources_v1.json').read_text())
    manifest = {'selection_lock_sha256': 'lock', 'rows': [{} for _ in range(135)]}
    plan = plan_physical_resources(manifest, resources, manifest_sha256='manifest')
    assert len(plan['shards']) == 17 and plan['max_concurrent_shards'] == 4
    assert plan['max_allocated_core_hours'] == 340
    resources['physical_replay']['max_concurrent_shards'] = 0
    with pytest.raises(ValueError, match='concurrency'):
        plan_physical_resources(manifest, resources, manifest_sha256='manifest')


def test_three_source_m9_submission_respects_array_cap(tmp_path):
    root = Path(__file__).resolve().parents[1]
    deployment = tmp_path/'deployment'; deployment.mkdir()
    (deployment/'src').symlink_to(root/'src', target_is_directory=True)
    (deployment/'DEPLOYED_GIT_COMMIT').write_text('a'*40)
    resources_path = root/'configs/research/exp_inv_013_resources_v1.json'
    resources_raw = resources_path.read_bytes()
    protocol = tmp_path/'protocol.json'; protocol.write_text('{}')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'selection_lock_sha256': 'lock',
        'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(), 'rows': [{}]*135}))
    plan = plan_physical_resources(json.loads(manifest.read_bytes()), json.loads(resources_raw),
                                   manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    plan['resources_sha256'] = hashlib.sha256(resources_raw).hexdigest()
    plan_path = tmp_path/'plan.json'; plan_path.write_text(json.dumps(plan))
    bin_dir = tmp_path/'bin'; bin_dir.mkdir()
    fake_sbatch = bin_dir/'sbatch'
    fake_sbatch.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$SBATCH_CAPTURE"\nprintf "12345\\n"\n')
    fake_sbatch.chmod(0o755)
    env = {**os.environ, 'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'],
           'PHC2D_ROOT': str(deployment), 'PHC2D_EXPECTED_COMMIT': 'a'*40,
           'PHC2D_MANIFEST': str(manifest), 'PHC2D_PROTOCOL': str(protocol),
           'PHC2D_RESOURCES': str(resources_path), 'PHC2D_RESOURCE_PLAN': str(plan_path),
           'PHC2D_OUTPUT_ROOT': str(tmp_path/'output'), 'PHC2D_RECEIPT_ROOT': str(tmp_path/'receipt'),
           'SBATCH_CAPTURE': str(tmp_path/'args')}
    result = subprocess.run(['bash', str(root/'jobs/bingxing_submit_inverse_012.sh')],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--array=0-16%4' in (tmp_path/'args').read_text()


def test_m9_dry_run_checks_manifest_and_budget_without_submitting(tmp_path):
    root = Path(__file__).resolve().parents[1]
    deployment = tmp_path/'deployment'; deployment.mkdir()
    (deployment/'src').symlink_to(root/'src', target_is_directory=True)
    (deployment/'DEPLOYED_GIT_COMMIT').write_text('a'*40)
    resources_path = root/'configs/research/exp_inv_012_resources_v1.json'
    resources_raw = resources_path.read_bytes()
    resources = json.loads(resources_raw)
    protocol = tmp_path/'protocol.json'; protocol.write_text('{}')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'selection_lock_sha256': 'lock',
        'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(), 'rows': [{}]*11}))
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plan = plan_physical_resources(json.loads(manifest.read_bytes()), resources, manifest_sha256=manifest_hash)
    plan['resources_sha256'] = hashlib.sha256(resources_raw).hexdigest()
    plan_path = tmp_path/'plan.json'; plan_path.write_text(json.dumps(plan))
    env = {**os.environ, 'PHC2D_ROOT': str(deployment), 'PHC2D_EXPECTED_COMMIT': 'a'*40,
           'PHC2D_MANIFEST': str(manifest), 'PHC2D_PROTOCOL': str(protocol),
           'PHC2D_RESOURCES': str(resources_path), 'PHC2D_RESOURCE_PLAN': str(plan_path),
           'PHC2D_OUTPUT_ROOT': str(tmp_path/'output'), 'PHC2D_RECEIPT_ROOT': str(tmp_path/'receipt'),
           'PHC2D_SUBMIT_DRY_RUN': '1'}
    script = root/'jobs/bingxing_submit_inverse_012.sh'
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'cases=11 shards=4 cpus=5 memory_mib=12288 wall_minutes=120' in result.stdout
    assert not (tmp_path/'receipt').exists() and not (tmp_path/'output').exists()
    bin_dir = tmp_path/'bin'; bin_dir.mkdir()
    fallback_python = bin_dir/'python3.11'
    fallback_python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    fallback_python.chmod(0o755)
    fake_sbatch = bin_dir/'sbatch'
    fake_sbatch.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$SBATCH_CAPTURE"\nprintf "12345\\n"\n')
    fake_sbatch.chmod(0o755)
    submit_env = {**env, 'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'],
                  'PHC2D_SUBMIT_PYTHON': str(tmp_path/'missing-python'),
                  'SBATCH_CAPTURE': str(tmp_path/'sbatch_args'), 'PHC2D_SUBMIT_DRY_RUN': '0'}
    submitted = subprocess.run(['bash', str(script)], env=submit_env, capture_output=True, text=True)
    assert submitted.returncode == 0, submitted.stderr
    assert submitted.stdout.strip() == '12345'
    arguments = (tmp_path/'sbatch_args').read_text().splitlines()
    assert '--array=0-3%4' in arguments and '--mem=12288M' in arguments
    assert arguments[arguments.index('-c') + 1] == '5'
    assert arguments[arguments.index('-t') + 1] == '120'
    repeated = subprocess.run(['bash', str(script)], env=submit_env, capture_output=True, text=True)
    assert repeated.returncode != 0 and 'already exists' in repeated.stderr
    plan['workers_per_shard'] = 4
    plan_path.write_text(json.dumps(plan))
    bad = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert bad.returncode != 0 and 'resource plan' in bad.stderr
