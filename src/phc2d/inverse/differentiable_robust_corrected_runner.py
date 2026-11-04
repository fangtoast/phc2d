"""Run paired robust designs with frozen scenario-specific physical feedback.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from phc2d.inverse.differentiable_robust_corrected import propose_corrected_robust_design
from phc2d.inverse.differentiable_selection import rank_robust_trajectory


def run_corrected_robust_trials(manifest: Mapping[str, Any], feedback: Mapping[str, Any], *,
                                manifest_sha256: str, feedback_sha256: str, source_id: int,
                                mode: str, source: Any, predictor: Any, runtime: Mapping[str, Any],
                                code_commit: str, cluster_id: str, output: Path,
                                radius: float, ridge: float) -> list[str]:
    """Run source/mode trials using only the frozen scenario stencil observations."""
    if (manifest.get('scope') != 'preregistered_robust_design'
            or feedback.get('scope') != 'robust_stencil_physical_observations'
            or feedback.get('campaign_manifest_sha256') != manifest_sha256
            or not manifest_sha256 or not feedback_sha256 or not code_commit or not cluster_id
            or mode not in ('nominal', 'mean', 'worst') or feedback['offsets'] != manifest['offsets']
            or radius != manifest['feedback']['radius'] or ridge != manifest['feedback']['ridge']):
        raise ValueError('invalid corrected robust campaign identity')
    sources = [item for item in manifest['sources'] if item['source_id'] == source_id]
    specs = [item for item in manifest['trials'] if item['source_id'] == source_id and item['mode'] == mode]
    if (len(sources) != 1 or source.parameter_hash != sources[0]['source_parameter_hash']
            or len(specs) != len(sources[0]['initials'])
            or {item['initial_index'] for item in specs} != {item['initial_index'] for item in sources[0]['initials']}):
        raise ValueError('corrected robust source or initial roster mismatch')
    output.mkdir(parents=True, exist_ok=True)
    completed = []
    for spec in specs:
        trial_id = spec['trial_id'] + '_corrected'
        path = output / f'{trial_id}.json'
        failure = output / f'{trial_id}.failure.json'
        identity = {'trial_id': trial_id, 'manifest_sha256': manifest_sha256, 'feedback_sha256': feedback_sha256,
                    'code_commit': code_commit, 'cluster_id': cluster_id}
        if failure.exists():
            raise FileExistsError(f'prior corrected robust failure must be reconciled: {trial_id}')
        if path.exists():
            old = json.loads(path.read_bytes())
            if any(old.get(key) != value for key, value in identity.items()):
                raise ValueError('existing corrected robust trial identity mismatch')
            completed.append(trial_id)
            continue
        if spec['status'] != 'ready':
            payload = {**identity, 'status': spec['status'], 'initial_index': spec['initial_index']}
        else:
            source_trial = f"source_{source_id:04d}_initial_{spec['initial_index']}_nominal"
            roles = [row for row in feedback['records'] if row['trial_id'] == source_trial]
            if len(roles) != 5 or len({row['role'] for row in roles}) != 5:
                raise ValueError('corrected robust physical stencil roster mismatch')
            bank = [scenario for row in roles if row['status'] == 'observed' for scenario in row['scenarios']]
            used_offsets = [0.0] if mode == 'nominal' else manifest['offsets']
            selected_bank = [row for row in bank if row['offset'] in used_offsets]
            if any(not any(row['offset'] == offset for row in selected_bank) for offset in used_offsets):
                payload = {**identity, 'status': 'physical_feedback_unavailable',
                           'initial_index': spec['initial_index'],
                           'feedback_role_statuses': {row['role']: row['status'] for row in roles}}
            else:
                cfg = dict(spec['config'], offsets=used_offsets, center=spec['config']['initial'],
                    radius=radius, ridge=ridge, surrogate_contract_sha256=feedback['surrogate_contract_sha256'],
                    scenario_reduction='mean' if mode == 'mean' else 'worst')
                started = time.monotonic()
                try:
                    result = propose_corrected_robust_design(source, predictor, selected_bank, config=cfg)
                    keys = tuple((int(i), str(component)) for i, component in cfg['coefficient_keys'])
                    diagnostic = rank_robust_trajectory(source, predictor, result['history'], offsets=manifest['offsets'],
                        coefficient_keys=keys, max_mode_radius=cfg['max_mode_radius'], raster_method='master_block_mean_v1',
                        source_resolution=cfg['source_resolution'], input_resolution=cfg['input_resolution'],
                        material_fraction_bounds=tuple(cfg['material_fraction_bounds']), lower_band=cfg['lower_band'],
                        target_frequency=cfg['target_frequency'])
                except Exception as exc:
                    with failure.open('x', encoding='utf8') as stream:
                        json.dump({**identity, 'error_type': type(exc).__name__, 'message': str(exc)}, stream)
                    raise
                payload = {**result, **identity, 'status': 'completed' if result['status'] == 'proposal_ready' else result['status'],
                    'mode': mode, 'source_id': source_id, 'source_parameter_hash': source.parameter_hash,
                    'initial_index': spec['initial_index'], 'source_sha256': cfg['source_sha256'],
                    'checkpoint_sha256': predictor.checkpoint_sha256, 'k_points_sha256': cfg['k_points_sha256'],
                    'k_points': predictor.k_points.detach().cpu().tolist(), 'config': cfg, 'runtime': dict(runtime),
                    'full_scenarios_diagnostic': diagnostic, 'elapsed_s': time.monotonic()-started,
                    'feedback_role_statuses': {row['role']: row['status'] for row in roles}}
        with path.open('x', encoding='utf8') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
        completed.append(trial_id)
    return completed


def main() -> None:
    from phc2d.inverse.differentiable_mvp import load_design_inputs

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'feedback', 'source', 'checkpoint', 'mode', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--source-id', required=True, type=int)
    parser.add_argument('--radius', required=True, type=float)
    parser.add_argument('--ridge', required=True, type=float)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    manifest_raw, feedback_raw = Path(args.manifest).read_bytes(), Path(args.feedback).read_bytes()
    manifest, feedback = json.loads(manifest_raw), json.loads(feedback_raw)
    trials = [row for row in manifest['trials'] if row['source_id'] == args.source_id
              and row['mode'] == args.mode and row['status'] == 'ready']
    if not trials:
        raise ValueError('no ready corrected robust trial for source and mode')
    source, predictor, runtime = load_design_inputs(trials[0]['config'], args.source, args.checkpoint, device=args.device)
    result = run_corrected_robust_trials(manifest, feedback, manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        feedback_sha256=hashlib.sha256(feedback_raw).hexdigest(), source_id=args.source_id, mode=args.mode,
        source=source, predictor=predictor, runtime=runtime, code_commit=os.environ['PHC2D_DEPLOYED_GIT_COMMIT'],
        cluster_id=os.environ['PHC2D_CLUSTER_ID'], output=Path(args.output), radius=args.radius, ridge=args.ridge)
    print(json.dumps({'completed_trial_ids': result}))


if __name__ == '__main__':
    main()
