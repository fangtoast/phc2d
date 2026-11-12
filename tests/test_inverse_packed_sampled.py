"""Contract tests for parallel sampled MPB screening.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest

from phc2d.inverse.packed_cases import run_packed_cases


def test_sampled_cases_share_capacity_guards_and_preserve_failures(tmp_path, monkeypatch):
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'rows':[{}, {}, {}, {}]}))
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    args=dict(mode='sampled',manifest=manifest,manifest_sha256=digest,output_root=tmp_path/'out',
        start=0,end=4,workers=2,cpu_limit=2,memory_budget_mib=2000,
        worker_memory_mib=500,setup_reserve_mib=1000)
    active=0
    peak=0
    lock=threading.Lock()
    seen=[]
    def fake_run(command,*,stdout,stderr,env,check):
        nonlocal active,peak
        assert command[2]=='phc2d.inverse.audit_mpb' and '--protocol' not in command
        index=int(command[command.index('--case-index')+1])
        with lock:
            active+=1
            peak=max(peak,active)
            seen.append(index)
        time.sleep(.03)
        with lock:
            active-=1
        return SimpleNamespace(returncode=1 if index==2 else 0)
    monkeypatch.setattr('phc2d.inverse.packed_cases.subprocess.run',fake_run)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK',raising=False)
    monkeypatch.delenv('SLURM_MEM_PER_NODE',raising=False)
    result=run_packed_cases(**args)
    assert peak==2 and sorted(seen)==[0,1,2,3]
    assert result['mode']=='sampled' and not result['all_succeeded']
    assert [x['returncode'] for x in result['cases']]==[0,0,1,0]
    with pytest.raises(FileExistsError):
        run_packed_cases(**args)


def test_sampled_case_rejects_protocol_or_overcommit_before_launch(tmp_path):
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'rows':[{}]}))
    args=dict(mode='sampled',manifest=manifest,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        output_root=tmp_path/'out',start=0,end=1,workers=2,cpu_limit=1,memory_budget_mib=1000,
        worker_memory_mib=200,setup_reserve_mib=200)
    with pytest.raises(ValueError,match='budget'):
        run_packed_cases(**args)
    with pytest.raises(ValueError,match='do not use'):
        run_packed_cases(**dict(args,workers=1,protocol=manifest,protocol_sha256='x'))
    assert not args['output_root'].exists()
