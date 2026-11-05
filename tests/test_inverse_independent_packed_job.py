"""Exercise source-packed dispatch and reject changed manifest identities."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def test_source_packed_worker_runs_registered_cells_and_checks_hashes(tmp_path):
    root = tmp_path/'root'
    (root/'jobs').mkdir(parents=True)
    (root/'jobs/paracloud_gpu_common.sh').write_text('export PHC2D_DEPLOYED_GIT_COMMIT=frozen\n')
    fakebin = tmp_path/'bin'
    fakebin.mkdir()
    fake = fakebin/'python'
    fake.write_text(f'''#!{sys.executable}
import os,sys
if len(sys.argv)>1 and sys.argv[1]=='-m':
    with open(os.environ['TRACE'],'a',encoding='utf8') as f: f.write(' '.join(sys.argv[1:])+'\\n')
else:
    os.execv({sys.executable!r},[{sys.executable!r},*sys.argv[1:]])
''')
    fake.chmod(0o755)
    matrix = tmp_path/'matrix'
    cells = []
    for name in ('target_00_seed_1', 'target_01_seed_1'):
        folder = matrix/name
        folder.mkdir(parents=True)
        raw = (name+'\n').encode()
        (folder/'manifest.json').write_bytes(raw)
        cells.append({'name': name, 'manifest_sha256': hashlib.sha256(raw).hexdigest()})
    dispatch = tmp_path/'dispatch.json'
    raw = json.dumps({'scope': 'independent_gpu_packed_dispatch', 'sources': [{'id': 100001}],
                      'cells': cells}).encode()
    dispatch.write_bytes(raw)
    trace = tmp_path/'trace'
    env = dict(os.environ, PATH=str(fakebin)+os.pathsep+os.environ['PATH'], TRACE=str(trace),
        PHC2D_ROOT=str(root), PHC2D_GPU_PLATFORM='n56', PHC2D_EXPECTED_COMMIT='frozen',
        PHC2D_CONTROLLER_COMMIT='controller', PHC2D_DISPATCH=str(dispatch),
        PHC2D_DISPATCH_SHA256=hashlib.sha256(raw).hexdigest(), PHC2D_MATRIX_ROOT=str(matrix),
        PHC2D_SOURCE_DIRECTORY=str(tmp_path/'sources'), PHC2D_CHECKPOINT=str(tmp_path/'checkpoint'),
        PHC2D_OUTPUT_ROOT=str(tmp_path/'out'), SLURM_ARRAY_TASK_ID='0')
    script = Path(__file__).resolve().parents[1]/'jobs/paracloud_inverse_independent_packed.sh'
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = trace.read_text().splitlines()
    assert len(lines) == 6
    assert all('--source-id 100001' in line for line in lines)
    assert [line.split('--method ')[1].split()[0] for line in lines] == ['adam', 'de', 'random']*2
    (matrix/'target_00_seed_1/manifest.json').write_text('changed')
    failed = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert failed.returncode != 0


def test_packed_submission_requires_controller_identity_and_single_receipt(tmp_path):
    root = tmp_path/'root'
    root.mkdir()
    (root/'DEPLOYED_GIT_COMMIT').write_text('frozen\n')
    fakebin = tmp_path/'bin'
    fakebin.mkdir()
    fake = fakebin/'sbatch'
    fake.write_text('#!/bin/sh\nprintf "12345\\n"\n')
    fake.chmod(0o755)
    worker = tmp_path/'worker.sh'
    worker.write_text('#!/bin/sh\nexit 0\n')
    dispatch = tmp_path/'dispatch.json'
    raw = json.dumps({'scope':'independent_gpu_packed_dispatch', 'variant':'two',
                      'controller_commit':'controller', 'sources':[{'id': i} for i in range(5)]}).encode()
    dispatch.write_bytes(raw)
    env = dict(os.environ, PATH=str(fakebin)+os.pathsep+os.environ['PATH'],
        PHC2D_ROOT=str(root), PHC2D_EXPECTED_COMMIT='frozen', PHC2D_CONTROLLER_COMMIT='controller',
        PHC2D_GPU_PLATFORM='n56', PHC2D_DISPATCH=str(dispatch), PHC2D_DISPATCH_SHA256=hashlib.sha256(raw).hexdigest(),
        PHC2D_MATRIX_ROOT=str(tmp_path/'matrix'), PHC2D_SOURCE_DIRECTORY=str(tmp_path/'sources'),
        PHC2D_CHECKPOINT=str(tmp_path/'checkpoint'), PHC2D_OUTPUT_ROOT=str(tmp_path/'out'),
        PHC2D_RECEIPT_ROOT=str(tmp_path/'receipts'), PHC2D_WORKER=str(worker),
        PHC2D_WORKER_SHA256=hashlib.sha256(worker.read_bytes()).hexdigest(), PHC2D_VARIANT='two')
    script = Path(__file__).resolve().parents[1]/'jobs/paracloud_submit_inverse_independent_packed.sh'
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'job_id=12345' in (tmp_path/'receipts/submission_two.txt').read_text()
    assert subprocess.run(['bash', str(script)], env=env, capture_output=True).returncode != 0
    env.update(PHC2D_CONTROLLER_COMMIT='different', PHC2D_RECEIPT_ROOT=str(tmp_path/'other_receipts'))
    assert subprocess.run(['bash', str(script)], env=env, capture_output=True).returncode != 0
