"""Test campaign array submission identity and duplicate protection."""
import hashlib
import os
from pathlib import Path
import subprocess


def test_campaign_submission_is_bounded_and_idempotent(tmp_path):
    root = tmp_path/'deployment'
    (root/'jobs').mkdir(parents=True)
    (root/'DEPLOYED_GIT_COMMIT').write_text('a'*40)
    (root/'jobs/paracloud_inverse_differentiable_campaign.sh').write_text('#!/bin/sh\n')
    sources = tmp_path/'sources'; sources.mkdir()
    checkpoint = tmp_path/'model.pt'; checkpoint.write_bytes(b'model')
    manifest = tmp_path/'manifest.json'; manifest.write_text('{"sources":[{"source_id":8},{"source_id":491},{"source_id":348}]}')
    bindir = tmp_path/'bin'; bindir.mkdir()
    sbatch = bindir/'sbatch'
    sbatch.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$SBATCH_CAPTURE"\nprintf "12345\\n"\n')
    sbatch.chmod(0o755)
    script = Path(__file__).resolve().parents[1]/'jobs/paracloud_submit_inverse_campaign.sh'
    env = {**os.environ, 'PATH': str(bindir) + os.pathsep + os.environ['PATH'],
           'PHC2D_ROOT': str(root), 'PHC2D_GPU_PLATFORM': 'n40', 'PHC2D_METHOD': 'adam',
           'PHC2D_MANIFEST': str(manifest), 'PHC2D_MANIFEST_SHA256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
           'PHC2D_SOURCE_DIRECTORY': str(sources), 'PHC2D_CHECKPOINT': str(checkpoint),
           'PHC2D_EXPECTED_COMMIT': 'a'*40, 'PHC2D_OUTPUT_ROOT': str(tmp_path/'out'),
           'PHC2D_RECEIPT_ROOT': str(tmp_path/'receipts'), 'SBATCH_CAPTURE': str(tmp_path/'capture')}
    dry = subprocess.run(['bash', str(script)], env={**env, 'PHC2D_SUBMIT_DRY_RUN': '1'}, capture_output=True, text=True)
    assert dry.returncode == 0 and not (tmp_path/'receipts').exists()
    submitted = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert submitted.returncode == 0 and submitted.stdout.strip() == '12345'
    assert '--array=0-2' in (tmp_path/'capture').read_text()
    assert 'job_id=12345' in (tmp_path/'receipts/submission_adam.txt').read_text()
    duplicate = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert duplicate.returncode != 0 and 'receipt exists' in duplicate.stderr
    wrong = subprocess.run(['bash', str(script)], env={**env, 'PHC2D_GPU_PLATFORM': 'n56'}, capture_output=True, text=True)
    assert wrong.returncode != 0 and 'takeover' in wrong.stderr
    takeover = {**env, 'PHC2D_GPU_PLATFORM': 'n56', 'PHC2D_N56_TAKEOVER': '1', 'PHC2D_SUBMIT_DRY_RUN': '1'}
    assert subprocess.run(['bash', str(script)], env=takeover, capture_output=True).returncode == 0
    feedback = {**env, 'PHC2D_GPU_PLATFORM': 'n56', 'PHC2D_METHOD': 'feedback_seed', 'PHC2D_SUBMIT_DRY_RUN': '1'}
    assert subprocess.run(['bash', str(script)], env=feedback, capture_output=True).returncode == 0
    feedback_round = {**feedback, 'PHC2D_METHOD': 'feedback_round'}
    assert subprocess.run(['bash', str(script)], env=feedback_round, capture_output=True).returncode == 0
    robust = {**feedback, 'PHC2D_METHOD': 'worst'}
    assert subprocess.run(['bash', str(script)], env=robust, capture_output=True).returncode == 0
    for method in ('robust_stencil', 'nominal_corrected', 'mean_corrected', 'worst_corrected'):
        assert subprocess.run(['bash', str(script)], env={**feedback, 'PHC2D_METHOD': method}, capture_output=True).returncode == 0
    manifest.write_text('{"sources":[{"source_id":51},{"source_id":52},{"source_id":53},{"source_id":54}]}')
    dynamic = {**robust, 'PHC2D_MANIFEST_SHA256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    check = subprocess.run(['bash', str(script)], env=dynamic, capture_output=True, text=True)
    assert check.returncode == 0 and 'sources=4' in check.stdout
    override = {**feedback, 'PHC2D_JOB_SCRIPT': str(root/'jobs/paracloud_inverse_differentiable_campaign.sh')}
    assert subprocess.run(['bash', str(script)], env=override, capture_output=True).returncode != 0
