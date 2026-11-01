"""Run one immutable materialized paired-search request.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def execute_paired_search_request(
    request_root: Path, *, manifest_sha256: str, request_index: int,
    output_root: Path, device: str,
) -> dict[str, Any]:
    """Execute a single identity-checked search without replacing prior output.

    The scheduler caller must authenticate deployment, experiment chronology,
    request manifest origin, resource reservations and terminal status. This
    function only runs the surrogate; no physical claim follows from success.
    """
    root = Path(request_root).resolve()
    output = Path(output_root).resolve()
    if (isinstance(request_index, bool) or not isinstance(request_index, int) or request_index < 0
            or not isinstance(device, str) or not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', device)
            or output == root or root in output.parents):
        raise ValueError('invalid request index, device or output location')
    if output.exists():
        raise FileExistsError('search output exists; reconcile before retry')
    manifest_raw = (root/'manifest.json').read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != manifest_sha256:
        raise ValueError('request manifest hash mismatch')
    manifest = json.loads(manifest_raw)
    records = manifest.get('requests')
    if (manifest.get('scope') != 'materialized_search_requests_not_executed'
            or not isinstance(records, list) or request_index >= len(records)):
        raise ValueError('invalid request manifest or index')
    record = records[request_index]
    sid, seed = record['source_index'], record['model_seed']
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (sid,seed)):
        raise ValueError('invalid request identity')
    source_name, config_name = f'sources/source_{sid:04d}.json', f'configs/source_{sid:04d}_model_{seed}.json'
    if record['source_file'] != source_name or record['config_file'] != config_name:
        raise ValueError('request file roster mismatch')
    source_path, config_path = root/source_name, root/config_name
    source_raw, config_raw = source_path.read_bytes(), config_path.read_bytes()
    if (hashlib.sha256(source_raw).hexdigest() != record['source_sha256']
            or hashlib.sha256(config_raw).hexdigest() != record['config_sha256']):
        raise ValueError('request source or configuration hash mismatch')
    config = json.loads(config_raw)
    if (Path(config['source']).resolve() != source_path.resolve()
            or config['checkpoint_sha256'] != record['checkpoint_sha256']
            or config['seeds'] != [record['search_seed']]):
        raise ValueError('request source, checkpoint or seed identity mismatch')
    checkpoint = Path(config['checkpoint'])
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != record['checkpoint_sha256']:
        raise ValueError('checkpoint hash mismatch')
    subprocess.run([sys.executable, '-m', 'phc2d.inverse.surrogate_design', '--config', str(config_path),
                    '--output-root', str(output), '--device', device], check=True)
    summary = (output/'summary.json').read_bytes()
    return {'scope':'surrogate_execution_not_physical_validation', 'manifest_sha256':manifest_sha256,
            'request_index':request_index, 'source_index':sid, 'model_seed':seed,
            'summary_sha256':hashlib.sha256(summary).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description='Execute one frozen paired inverse search request')
    parser.add_argument('--request-root', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--request-index', type=int, required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    result = execute_paired_search_request(Path(args.request_root), manifest_sha256=args.manifest_sha256,
        request_index=args.request_index, output_root=Path(args.output_root), device=args.device)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
