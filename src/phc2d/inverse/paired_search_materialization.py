"""Materialize and audit immutable paired inverse-search request files.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def _expected(batch: Mapping[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    if batch.get('scope') != 'frozen_search_requests_not_executed' or not batch.get('requests'):
        raise ValueError('invalid search request batch')
    files: dict[str, bytes] = {}
    records = []
    seen = set()
    source_paths: dict[int, str] = {}
    for request in batch['requests']:
        sid, seed = request['source_index'], request['model_seed']
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (sid, seed))
                or (sid, seed) in seen):
            raise ValueError('invalid or duplicate source/model identity')
        seen.add((sid, seed))
        source_name = f'sources/source_{sid:04d}.json'
        config_name = f'configs/source_{sid:04d}_model_{seed}.json'
        source = _bytes(request['source'])
        source_hash = hashlib.sha256(source).hexdigest()
        if (source_hash != request['source_sha256']
                or not request['config']['source'].endswith(f'/source_{sid:04d}.json')
                or request['config']['seeds'] != [request['search_seed']]):
            raise ValueError('source or seed identity mismatch')
        if source_name in files and files[source_name] != source:
            raise ValueError('source differs across model seeds')
        if sid in source_paths and source_paths[sid] != request['config']['source']:
            raise ValueError('source path differs across model seeds')
        source_paths[sid] = request['config']['source']
        files[source_name] = source
        raw_config = _bytes(request['config'])
        files[config_name] = raw_config
        records.append({'source_index':sid, 'stratum':request['stratum'], 'model_seed':seed,
                        'search_seed':request['search_seed'], 'source_file':source_name,
                        'source_sha256':source_hash, 'config_file':config_name,
                        'config_sha256':hashlib.sha256(raw_config).hexdigest(),
                        'checkpoint_sha256':request['config']['checkpoint_sha256']})
    identity = {k:batch[k] for k in ('config_sha256','source_manifest_sha256','source_audit_sha256','selection_sha256')}
    identity.update(scope='materialized_search_requests_not_executed', requests=records)
    return files, identity


def materialize_paired_search_requests(batch: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    """Write a new request directory; retain any partial output after failure."""
    files, identity = _expected(batch)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    (root/'sources').mkdir()
    (root/'configs').mkdir()
    for name, raw in files.items():
        with (root/name).open('xb') as stream:
            stream.write(raw)
    manifest_raw = _bytes(identity)
    with (root/'manifest.json').open('xb') as stream:
        stream.write(manifest_raw)
    return {'scope':'request_files_created_not_executed', 'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),
            'request_count':len(identity['requests']), 'source_count':sum(name.startswith('sources/') for name in files)}


def audit_paired_search_requests(batch: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    """Compare all materialized bytes to an independently rebuilt batch."""
    files, identity = _expected(batch)
    root = Path(output_root)
    manifest_raw = (root/'manifest.json').read_bytes()
    if manifest_raw != _bytes(identity):
        raise ValueError('materialized request manifest mismatch')
    expected_names = set(files) | {'manifest.json'}
    actual_names = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual_names != expected_names:
        raise ValueError('materialized request file roster mismatch')
    for name, expected in files.items():
        if (root/name).read_bytes() != expected:
            raise ValueError(f'materialized request file mismatch: {name}')
    return {'scope':'request_files_audited_not_executed', 'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),
            'request_count':len(identity['requests']), 'source_count':sum(name.startswith('sources/') for name in files)}
