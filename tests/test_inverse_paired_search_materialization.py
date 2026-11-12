"""Check request materialization and deterministic corruption detection.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import runpy
import copy
from pathlib import Path

import pytest

from phc2d.inverse.paired_search_contract import build_paired_search_requests
from phc2d.inverse.paired_search_materialization import materialize_paired_search_requests, audit_paired_search_requests


def test_request_materialization_and_complete_byte_audit(tmp_path):
    fixture = runpy.run_path(str(Path(__file__).with_name('test_inverse_paired_search_contract.py')))['fixture_contract']
    batch = build_paired_search_requests(**fixture())
    output = tmp_path/'requests'
    created = materialize_paired_search_requests(batch, output)
    checked = audit_paired_search_requests(batch, output)
    assert created['manifest_sha256'] == checked['manifest_sha256']
    assert checked['request_count'] == 9 and checked['source_count'] == 3
    with pytest.raises(FileExistsError):
        materialize_paired_search_requests(batch, output)
    config = output/'configs/source_0000_model_0.json'
    config.write_text('{}')
    with pytest.raises(ValueError, match='request file mismatch'):
        audit_paired_search_requests(batch, output)
    altered = copy.deepcopy(batch)
    altered['requests'][1]['config']['source'] = '/different/source_0000.json'
    with pytest.raises(ValueError, match='source path differs'):
        materialize_paired_search_requests(altered, tmp_path/'bad')
