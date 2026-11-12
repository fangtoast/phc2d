"""Check packed feedback cells retain their frozen inputs and source identity."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def test_feedback_worker_rejects_changed_arms(tmp_path):
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
else: os.execv({sys.executable!r},[{sys.executable!r},*sys.argv[1:]])
''')
    fake.chmod(0o755)
    matrix, feedback = tmp_path/'matrix', tmp_path/'feedback'
    cells = []
    for name in ('target_01_seed_1', 'target_02_seed_1'):
        (matrix/name).mkdir(parents=True)
        (feedback/name).mkdir(parents=True)
        manifest, arms = name.encode(), (name+'-arms').encode()
        (matrix/name/'manifest.json').write_bytes(manifest)
        (feedback/name/'round1_arms.json').write_bytes(arms)
        cells.append({'name':name, 'manifest_sha256':hashlib.sha256(manifest).hexdigest(),
                      'arms_sha256':hashlib.sha256(arms).hexdigest()})
    dispatch = tmp_path/'dispatch.json'
    raw = json.dumps({'scope':'independent_feedback_packed_dispatch','round':1,
                      'sources':[{'id':100001}],'cells':cells}).encode()
    dispatch.write_bytes(raw)
    campaign = tmp_path/'campaign.json'
    campaign.write_bytes(b'campaign')
    trace = tmp_path/'trace'
    env = dict(os.environ, PATH=str(fakebin)+os.pathsep+os.environ['PATH'], TRACE=str(trace),
        PHC2D_ROOT=str(root), PHC2D_GPU_PLATFORM='n56', PHC2D_EXPECTED_COMMIT='frozen',
        PHC2D_DISPATCH=str(dispatch), PHC2D_DISPATCH_SHA256=hashlib.sha256(raw).hexdigest(),
        PHC2D_MATRIX_ROOT=str(matrix), PHC2D_SOURCE_DIRECTORY=str(tmp_path/'sources'),
        PHC2D_CHECKPOINT=str(tmp_path/'checkpoint'), PHC2D_OUTPUT_ROOT=str(feedback),
        PHC2D_FEEDBACK_CONFIG=str(campaign), PHC2D_FEEDBACK_CONFIG_SHA256=hashlib.sha256(b'campaign').hexdigest(),
        SLURM_ARRAY_TASK_ID='0')
    worker = Path(__file__).resolve().parents[1]/'jobs/paracloud_inverse_feedback_packed.sh'
    result = subprocess.run(['bash',str(worker)],env=env,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    lines = trace.read_text().splitlines()
    assert len(lines)==2 and all('--source-id 100001' in line and '--round 1' in line for line in lines)
    (feedback/'target_02_seed_1/round1_arms.json').write_bytes(b'changed')
    assert subprocess.run(['bash',str(worker)],env=env,capture_output=True).returncode != 0
