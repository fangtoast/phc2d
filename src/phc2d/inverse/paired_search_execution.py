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
from pathlib import Path, PurePosixPath
from typing import Any


def inspect_nominal_static_request(request_root: Path, *, manifest_sha256: str, source_index: int,
                                   expected_seeds: list[int], checkpoint_sha256: str) -> dict[str, Any]:
    """Inspect one two-trajectory draft; this does not admit or run a job.

    The caller authenticates the draft manifest and scientific contract. Remote
    paths are checked lexically here; checkpoint bytes and deployment must be
    checked on the execution host before any admitted search.
    """
    from phc2d.data.full_cell_spec import FullCellStructureSpec

    root = Path(request_root)
    if (isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0
            or not isinstance(expected_seeds, list) or len(expected_seeds) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in expected_seeds)
            or len(set(expected_seeds)) != 2 or not isinstance(checkpoint_sha256, str)
            or not re.fullmatch(r'[0-9a-f]{64}', checkpoint_sha256)):
        raise ValueError('invalid source, seed or checkpoint identity')
    raw = (root/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError('request manifest hash mismatch')
    manifest = json.loads(raw)
    if (manifest.get('scope') != 'static_request_materialization_not_executed'
            or manifest.get('status') != 'draft_not_execution_lock'
            or not isinstance(manifest.get('records'), list)
            or not all(isinstance(row, dict) for row in manifest['records'])):
        raise ValueError('invalid nominal draft manifest')
    matches = [row for row in manifest['records'] if row.get('source_index') == source_index]
    if len(matches) != 1:
        raise ValueError('source identity missing or duplicated')
    record = matches[0]
    source_name, config_name = f'sources/source_{source_index:04d}.json', f'configs/source_{source_index:04d}.json'
    if record.get('source_file') != source_name or record.get('config_file') != config_name:
        raise ValueError('request file roster mismatch')
    source_raw, config_raw = (root/source_name).read_bytes(), (root/config_name).read_bytes()
    if (hashlib.sha256(source_raw).hexdigest() != record.get('source_sha256')
            or hashlib.sha256(config_raw).hexdigest() != record.get('config_sha256')):
        raise ValueError('request source or configuration hash mismatch')
    source = FullCellStructureSpec.from_mapping(json.loads(source_raw))
    config = json.loads(config_raw)
    remote = manifest.get('remote_request_directory')
    if (not isinstance(remote, str) or not PurePosixPath(remote).is_absolute()
            or '..' in PurePosixPath(remote).parts
            or config.get('source') != str(PurePosixPath(remote)/source_name)
            or source.parameter_hash != record.get('parameter_hash')
            or config.get('checkpoint_sha256') != checkpoint_sha256
            or config.get('seeds') != expected_seeds
            or config.get('search_options', {}).get('perturbation_offsets') != [0.0]):
        raise ValueError('nominal source, checkpoint, seeds or offset mismatch')
    return {'scope': 'draft_inspection_not_execution_lock', 'source_index': source_index,
            'manifest_sha256': manifest_sha256, 'source_sha256': record['source_sha256'],
            'config_sha256': record['config_sha256'], 'parameter_hash': source.parameter_hash,
            'checkpoint_sha256': checkpoint_sha256, 'seeds': list(expected_seeds)}


def audit_nominal_static_request(request_root: Path, *, manifest_sha256: str, source_index: int,
                                 pilot_config_bytes: bytes, prior_config_bytes: bytes,
                                 expected_source_sha256: str, expected_checkpoint_path: str) -> dict[str, Any]:
    """Rebuild the scientific request from authenticated pilot and prior configs.

    The caller independently authenticates the manifest, source identity,
    checkpoint path, and original physical evidence. This audit is not a launch
    lock and does not verify checkpoint bytes or scheduler resources.
    """
    import yaml
    from phc2d.data.full_cell_spec import FullCellStructureSpec
    from phc2d.inverse.paired_search_contract import build_nominal_static_request

    root = Path(request_root)
    manifest = json.loads((root/'manifest.json').read_bytes())
    if (hashlib.sha256(pilot_config_bytes).hexdigest() != manifest.get('pilot_config_sha256')
            or hashlib.sha256(prior_config_bytes).hexdigest() != manifest.get('prior_config_sha256')):
        raise ValueError('scientific config identity mismatch')
    pilot = yaml.safe_load(pilot_config_bytes)
    inspected = inspect_nominal_static_request(root, manifest_sha256=manifest_sha256,
        source_index=source_index, expected_seeds=pilot['search']['seeds_by_round'],
        checkpoint_sha256=pilot['checkpoint_sha256'])
    if inspected['source_sha256'] != expected_source_sha256:
        raise ValueError('independent source identity mismatch')
    source_name = f'sources/source_{source_index:04d}.json'
    config_name = f'configs/source_{source_index:04d}.json'
    source = FullCellStructureSpec.from_mapping(json.loads((root/source_name).read_bytes()))
    config = json.loads((root/config_name).read_bytes())
    rebuilt = build_nominal_static_request(pilot_config_bytes=pilot_config_bytes, prior_config_bytes=prior_config_bytes,
        source=source, source_index=source_index,
        source_path=str(PurePosixPath(manifest['remote_request_directory'])/source_name),
        checkpoint_path=expected_checkpoint_path)
    if config != rebuilt:
        raise ValueError('nominal scientific request differs from frozen configs')
    return {**inspected, 'scope': 'draft_scientific_audit_not_execution_lock',
            'pilot_config_sha256': manifest['pilot_config_sha256'],
            'prior_config_sha256': manifest['prior_config_sha256']}


def execute_budgeted_nominal_static_request(request_root: Path, *, manifest_sha256: str, source_index: int,
                                            pilot_config_bytes: bytes, prior_config_bytes: bytes,
                                            expected_source_sha256: str, checkpoint_path: str,
                                            device: str, output_root: Path) -> dict[str, Any]:
    """Execute one audited nominal request; the caller must enforce admission.

    This primitive does not authenticate a prelaunch lock, resource quota,
    scheduler state or experiment chronology. It must not be used as a job
    controller or substitute for the independent admission audit.
    """
    from phc2d.data.full_cell_spec import FullCellStructureSpec
    from phc2d.inverse.frozen_band_adapter import FrozenBandAdapter
    from phc2d.inverse.surrogate_design import run_budgeted_nominal_static_design

    root, output = Path(request_root).resolve(), Path(output_root).resolve()
    if (not isinstance(device, str) or not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', device)
            or output == root or root in output.parents):
        raise ValueError('invalid device or output location')
    if output.exists():
        raise FileExistsError('search output exists; reconcile before retry')
    inspected = audit_nominal_static_request(root, manifest_sha256=manifest_sha256, source_index=source_index,
        pilot_config_bytes=pilot_config_bytes, prior_config_bytes=prior_config_bytes,
        expected_source_sha256=expected_source_sha256, expected_checkpoint_path=checkpoint_path)
    source_path = root/f'sources/source_{source_index:04d}.json'
    config_path = root/f'configs/source_{source_index:04d}.json'
    config_raw, source_raw = config_path.read_bytes(), source_path.read_bytes()
    if (hashlib.sha256(config_raw).hexdigest() != inspected['config_sha256']
            or hashlib.sha256(source_raw).hexdigest() != expected_source_sha256):
        raise ValueError('request files changed after audit')
    config = json.loads(config_raw)
    if Path(config['source']).resolve() != source_path.resolve():
        raise ValueError('deployed source path differs from frozen request')
    predictor = FrozenBandAdapter(checkpoint_path, sha256=config['checkpoint_sha256'], device=device)
    source = FullCellStructureSpec.from_mapping(json.loads(source_raw))
    options = dict(config['search_options'])
    if (predictor.input_shape != (options['input_resolution'], options['input_resolution'])
            or len(predictor.output_shape) != 2 or predictor.output_shape[1] <= options['lower_band']):
        raise ValueError('checkpoint shape differs from nominal request')
    options['coefficient_keys'] = tuple(tuple(key) for key in options['coefficient_keys'])
    options['bounds'] = tuple(tuple(bound) for bound in options['bounds'])
    options['material_fraction_bounds'] = tuple(options['material_fraction_bounds'])
    options['perturbation_offsets'] = tuple(options['perturbation_offsets'])
    provenance = {'config_sha256': hashlib.sha256(config_raw).hexdigest(),
        'source_sha256': hashlib.sha256(source_raw).hexdigest(), 'checkpoint_sha256': predictor.checkpoint_sha256,
        'model_name': predictor.model_name, 'k_points_sha256': predictor.k_points_sha256,
        'device': device, 'request_manifest_sha256': manifest_sha256}
    result = run_budgeted_nominal_static_design(source, predictor, seeds=config['seeds'], search_options=options,
        output_root=output, provenance=provenance, config_bytes=config_raw)
    return {'scope': 'budgeted_nominal_search_not_execution_lock', 'source_index': source_index,
            'request_manifest_sha256': manifest_sha256, 'request_config_sha256': inspected['config_sha256'],
            'trajectories': result['trajectories']}


def lock_budgeted_nominal_static_candidates(request_root: Path, *, manifest_sha256: str, source_index: int,
                                             pilot_config_bytes: bytes, prior_config_bytes: bytes,
                                             expected_source_sha256: str, checkpoint_path: str,
                                             output_root: Path, selection_path: Path) -> dict[str, Any]:
    """Audit both saved trajectories, then write one source's static selection.

    Scheduler identity, independent request audit, chronology and all other
    sources remain caller-owned. This is not a campaign-wide execution lock.
    """
    from phc2d.data.full_cell_spec import FullCellStructureSpec
    from phc2d.inverse.frozen_band_adapter import audit_prediction_archive_against_calls
    from phc2d.inverse.surrogate_design import audit_nominal_prediction_archive, audit_surrogate_design
    from phc2d.inverse.trace_selection import lock_static_nominal_candidates
    import yaml

    request_root, output_root, selection_path = Path(request_root), Path(output_root), Path(selection_path)
    request = audit_nominal_static_request(request_root, manifest_sha256=manifest_sha256, source_index=source_index,
        pilot_config_bytes=pilot_config_bytes, prior_config_bytes=prior_config_bytes,
        expected_source_sha256=expected_source_sha256, expected_checkpoint_path=checkpoint_path)
    config_raw = (request_root/f'configs/source_{source_index:04d}.json').read_bytes()
    source_raw = (request_root/f'sources/source_{source_index:04d}.json').read_bytes()
    if (hashlib.sha256(config_raw).hexdigest() != request['config_sha256']
            or hashlib.sha256(source_raw).hexdigest() != expected_source_sha256):
        raise ValueError('request changed after audit')
    config = json.loads(config_raw)
    source = FullCellStructureSpec.from_mapping(json.loads(source_raw))
    traces, evidence = [], []
    for seed in config['seeds']:
        seed_path = output_root/f'config_{seed}.json'
        seed_raw = seed_path.read_bytes()
        expected_config = {**config, 'seeds': [seed]}
        if json.loads(seed_raw) != expected_config:
            raise ValueError('trajectory config differs from frozen request')
        run = output_root/f'trajectory_{seed}'
        summary = json.loads((run/'summary.json').read_bytes())
        if (summary['provenance'].get('parent_config_sha256') != hashlib.sha256(config_raw).hexdigest()
                or summary['provenance'].get('request_manifest_sha256') != manifest_sha256):
            raise ValueError('trajectory parent configuration differs')
        audit_surrogate_design(run, config_bytes=seed_raw, source_bytes=source_raw,
                               checkpoint_sha256=config['checkpoint_sha256'])
        trace_raw = (run/f'seed_{seed}'/'search.json').read_bytes()
        trace = json.loads(trace_raw)
        cost_raw = (output_root/f'cost_{seed}.json').read_bytes()
        cost = json.loads(cost_raw)
        if cost.get('seed') != seed or cost.get('max_inferences') != config['search_options']['evaluation_budget']:
            raise ValueError('trajectory cost identity mismatch')
        archive = output_root/f'predictions_{seed}.npz'
        audit_prediction_archive_against_calls(output_root/f'inference_{seed}', archive, cost)
        audit_nominal_prediction_archive(trace=trace, options=config['search_options'], cost=cost, archive_path=archive)
        traces.append(trace['evaluations'])
        evidence.append({'seed': seed, 'trace_sha256': hashlib.sha256(trace_raw).hexdigest(),
            'cost_sha256': hashlib.sha256(cost_raw).hexdigest(),
            'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()})
    selected = lock_static_nominal_candidates(source, traces, raster_resolution=config['search_options']['source_resolution'],
        candidates=yaml.safe_load(pilot_config_bytes)['search']['candidate_physical_attempts_per_method'],
        evaluations_per_trace=config['search_options']['evaluation_budget'])
    result = {'scope': 'one_source_static_selection_not_execution_lock', 'source_index': source_index,
              'request_manifest_sha256': manifest_sha256, 'source_sha256': expected_source_sha256,
              'request_config_sha256': request['config_sha256'], 'evidence': evidence, 'selection': selected}
    with selection_path.open('xb') as stream:
        stream.write(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8'))
    return result


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
