"""Verify submission receipts prevent duplicate inverse job submissions.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import os
import json
from pathlib import Path
import subprocess


def test_submission_receipt_and_platform_assignment(tmp_path):
    root = tmp_path/'deployment'
    (root/'jobs').mkdir(parents=True)
    (root/'configs/research').mkdir(parents=True)
    (root/'DEPLOYED_GIT_COMMIT').write_text('a'*40)
    (root/'configs/research/exp_inv_012_origin_v1.json').write_text('{}')
    (root/'jobs/paracloud_inverse_differentiable_mvp.sh').write_text('#!/bin/sh\n')
    source, checkpoint = tmp_path/'source.json', tmp_path/'model.pt'
    source.write_text('{}'); checkpoint.write_bytes(b'checkpoint')
    bin_dir = tmp_path/'bin'; bin_dir.mkdir()
    sbatch = bin_dir/'sbatch'
    sbatch.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$SBATCH_CAPTURE"\nprintf "12345\\n"\n')
    sbatch.chmod(0o755)
    script = Path(__file__).resolve().parents[1]/'jobs/paracloud_submit_inverse_differentiable.sh'
    env = {**os.environ, 'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'],
           'PHC2D_ROOT': str(root), 'PHC2D_GPU_PLATFORM': 'n40', 'PHC2D_ROLE': 'preflight',
           'PHC2D_SOURCE': str(source), 'PHC2D_CHECKPOINT': str(checkpoint),
           'PHC2D_EXPECTED_COMMIT': 'a'*40, 'PHC2D_BATCH_ROOT': str(tmp_path/'batch'),
           'SBATCH_CAPTURE': str(tmp_path/'sbatch_args')}
    dry = subprocess.run(['bash', str(script)], env={**env, 'PHC2D_SUBMIT_DRY_RUN': '1'},
                         capture_output=True, text=True)
    assert dry.returncode == 0 and 'platform=n40 role=preflight' in dry.stdout
    assert not (tmp_path/'batch').exists() and not (tmp_path/'sbatch_args').exists()
    first = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert first.returncode == 0 and first.stdout.strip() == '12345'
    resources = json.loads((Path(__file__).resolve().parents[1]/'configs/research/exp_inv_012_resources_v1.json').read_text())
    gpu = resources['gpu_design']
    arguments = (tmp_path/'sbatch_args').read_text().splitlines()
    assert f"--gres=gpu:{gpu['gpus_per_job']}" in arguments
    assert arguments[arguments.index('-c') + 1] == str(gpu['cpus_per_job'])
    assert arguments[arguments.index('-t') + 1] == '00:30:00'
    assert gpu['wall_seconds_per_job'] == 1800
    assert 'job_id=12345' in (tmp_path/'batch/submission_preflight.txt').read_text()
    second = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert second.returncode != 0 and 'already exists' in second.stderr
    wrong = subprocess.run(['bash', str(script)], env={**env, 'PHC2D_ROLE': 'de'}, capture_output=True, text=True)
    assert wrong.returncode != 0 and 'not assigned' in wrong.stderr
