"""Predict frozen local feedback stencils under manufacturing scenarios.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids, predict_perturbed_gap


def predict_stencil_scenarios(stencil: Mapping[str, Any], source: Any, predictor: Any, *,
                              base: Mapping[str, Any], offsets: Sequence[float]) -> dict[str, Any]:
    """Save hard and smooth scenario spectra before obtaining physical feedback."""
    import torch
    from phc2d.data.differentiable_fourier import fourier_material_weight
    from phc2d.inverse.differentiable_morphology import periodic_soft_offset
    from phc2d.inverse.differentiable_objective import smooth_band_edges

    values = tuple(float(value) for value in offsets)
    if (len(values) != 3 or len(set(values)) != 3 or 0.0 not in values or not np.isfinite(values).all()
            or not any(x < 0 for x in values) or not any(x > 0 for x in values)
            or stencil.get('source_parameter_hash') != source.parameter_hash):
        raise ValueError('invalid stencil source or manufacturing scenarios')
    keys = tuple((int(i), str(component)) for i, component in base['coefficient_keys'])
    reference = torch.tensor([source.parameters['modes'][i][component] for i, component in keys]
                             + [source.parameters['threshold']], dtype=torch.float64, device=predictor.device)
    scales = torch.as_tensor(base['scales'], dtype=reference.dtype, device=reference.device)
    resolution, inputs = base['source_resolution'], base['input_resolution']
    if scales.shape != reference.shape or inputs <= 0 or resolution % inputs:
        raise ValueError('invalid stencil geometry resolution or scale')
    factor, pair, beta = resolution // inputs, base['lower_band'], float(base['beta'])
    rows = []
    for record in stencil['records']:
        candidate = record['candidate']
        if candidate is None or candidate['status'] not in ('surrogate_only', 'duplicate_geometry'):
            rows.append({'label': record['label'], 'status': record.get('status',
                         'invalid_candidate' if candidate is None else candidate['status'])})
            continue
        point = torch.as_tensor(record['point'], dtype=reference.dtype, device=reference.device)
        if point.shape != reference.shape or not torch.isfinite(point).all():
            raise ValueError('invalid stencil normalized point')
        structure = FullCellStructureSpec.from_mapping(candidate['structure'])
        grids = periodic_perturbation_grids(structure, resolution=resolution, offsets=values)
        nominal_hash = hashlib.sha256(np.ascontiguousarray(grids[values.index(0.0)], dtype=np.uint8).tobytes()).hexdigest()
        if structure.parameter_hash != candidate['parameter_hash'] or nominal_hash != candidate['master_grid_sha256']:
            raise ValueError('stencil candidate geometry identity mismatch')
        hard = predict_perturbed_gap(grids, predictor, input_resolution=inputs, lower_band=pair,
                                     target_frequency=base['target_frequency'])
        item = {'label': record['label'], 'point': record['point'], 'parameter_hash': candidate['parameter_hash'],
                'hard': hard}
        if hard['status'] != 'surrogate_only':
            rows.append({**item, 'status': hard['status']})
            continue
        with torch.no_grad():
            weight = fourier_material_weight(source, reference + scales*point, coefficient_keys=keys,
                master_resolution=resolution, output_resolution=resolution, temperature=base['temperature'])
            scenarios = [periodic_soft_offset(weight, offset=value).reshape(inputs, factor, inputs, factor).mean(dim=(1, 3))
                         for value in values]
            bands = predictor.predict_tensor(torch.stack(scenarios))
            smooth = smooth_band_edges(bands, lower_band=pair, beta=beta).detach().cpu().numpy()
            raw = bands.detach().cpu().numpy()
        if not np.isfinite(raw).all() or not np.isfinite(smooth).all():
            raise ValueError('nonfinite soft stencil spectrum')
        hard_bands = np.asarray(hard['bands'], dtype=np.float64)
        exact = np.stack((raw[:, :, pair-1].max(axis=1), raw[:, :, pair].min(axis=1)), axis=1)
        hard_edges = np.stack((hard_bands[:, :, pair-1].max(axis=1), hard_bands[:, :, pair].min(axis=1)), axis=1)
        rows.append({**item, 'status': 'computed', 'soft_bands': raw.tolist(),
                     'soft_smooth_edges': smooth.tolist(), 'soft_exact_edges': exact.tolist(),
                     'hard_edges': hard_edges.tolist()})
    return {'scope': 'robust_stencil_surrogate_only', 'source_parameter_hash': source.parameter_hash,
            'initial': stencil['initial'], 'offsets': list(values), 'records': rows}


def main() -> None:
    from phc2d.inverse.differentiable_mvp import load_design_inputs

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'source', 'checkpoint', 'stencil-directory', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--source-id', required=True, type=int)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    raw = Path(args.manifest).read_bytes()
    manifest = json.loads(raw)
    if manifest.get('scope') != 'preregistered_robust_design':
        raise ValueError('preregistered robust campaign required')
    sources = [row for row in manifest['sources'] if row['source_id'] == args.source_id]
    trials = [row for row in manifest['trials'] if row['source_id'] == args.source_id and row['status'] == 'ready']
    if len(sources) != 1 or not trials:
        raise ValueError('source is absent or has no valid initial')
    source, predictor, runtime = load_design_inputs(trials[0]['config'], args.source, args.checkpoint, device=args.device)
    if source.parameter_hash != sources[0]['source_parameter_hash']:
        raise ValueError('robust stencil source identity mismatch')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for initial in sources[0]['initials']:
        index = initial['initial_index']
        path = Path(args.stencil_directory)/f'source_{args.source_id:04d}_initial_{index}.json'
        output = root/path.name
        if output.exists():
            raise FileExistsError('existing robust stencil must be reconciled before retry')
        raw_stencil = path.read_bytes()
        stencil = json.loads(raw_stencil)
        if (stencil['source_id'] != args.source_id or stencil['initial_index'] != index
                or stencil['design_manifest_sha256'] != manifest['parent_manifest_sha256']
                or stencil['checkpoint_sha256'] != predictor.checkpoint_sha256
                or stencil['k_points_sha256'] != predictor.k_points_sha256
                or stencil['status'] != ('prepared' if initial['status'] == 'ready' else initial['status'])
                or (initial['status'] == 'ready' and stencil['initial'] != initial['initial'])):
            raise ValueError('nominal stencil does not match frozen robust campaign')
        if initial['status'] == 'ready':
            payload = predict_stencil_scenarios(stencil, source, predictor, base=trials[0]['config'],
                                               offsets=manifest['offsets'])
        else:
            payload = {'scope': 'robust_stencil_surrogate_only', 'status': initial['status'], 'records': []}
        payload.update(source_id=args.source_id, initial_index=index,
            manifest_sha256=hashlib.sha256(raw).hexdigest(), stencil_sha256=hashlib.sha256(raw_stencil).hexdigest(),
            checkpoint_sha256=predictor.checkpoint_sha256, k_points_sha256=predictor.k_points_sha256,
            runtime=runtime, code_commit=os.environ['PHC2D_DEPLOYED_GIT_COMMIT'],
            cluster_id=os.environ['PHC2D_CLUSTER_ID'])
        with output.open('x', encoding='utf8') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
