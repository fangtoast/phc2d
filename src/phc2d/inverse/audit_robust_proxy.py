"""Score frozen inverse candidates using a sampled surrogate common gap.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.audit_refine import read_checked_json
from phc2d.inverse.frozen_band_adapter import FrozenBandAdapter
from phc2d.inverse.perturbation_screen import compare_predictor_devices, periodic_perturbation_grids, predict_perturbed_gap
from phc2d.inverse.robust_gap import common_gap


def validate_frozen_proxy_predictions(
    *, manifest_bytes: bytes, manifest_sha256: str, protocol_bytes: bytes, protocol_sha256: str,
    proxy: Mapping[str, Any], checkpoint_sha256: str, k_points_sha256: str,
    checkpoint_output_shape: tuple[int, int],
) -> dict[str, Any]:
    """Rebuild input rasters and scores before any candidate physical results.

    The caller independently verifies checkpoint identity and output metadata.
    A nonfinite prediction whose raw spectrum is absent remains explicitly
    unscoreable; its specific failure cause cannot be independently replayed.
    This check establishes neither model accuracy nor physical acceptance.
    """
    if (hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256
            or hashlib.sha256(protocol_bytes).hexdigest() != protocol_sha256):
        raise ValueError('manifest or protocol digest mismatch')
    manifest, protocol = json.loads(manifest_bytes), json.loads(protocol_bytes)
    points = np.ascontiguousarray(manifest['k_points'], dtype=np.float32)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or hashlib.sha256(points.tobytes()).hexdigest() != k_points_sha256):
        raise ValueError('checkpoint k-point identity differs')
    if (not isinstance(checkpoint_output_shape, tuple) or len(checkpoint_output_shape) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in checkpoint_output_shape)
            or checkpoint_output_shape[0] != len(points)
            or not 1 <= manifest['lower_band'] < checkpoint_output_shape[1]):
        raise ValueError('invalid checkpoint output shape or band')
    offsets = protocol['offsets']
    if (proxy.get('status') != 'exploratory_surrogate_common_gap_only'
            or proxy.get('manifest_sha256') != manifest_sha256 or proxy.get('protocol_sha256') != protocol_sha256
            or proxy.get('checkpoint_sha256') != checkpoint_sha256 or proxy.get('k_points_sha256') != k_points_sha256
            or proxy.get('offsets') != offsets):
        raise ValueError('proxy prediction provenance differs')
    entries = manifest['rows']
    rows = proxy.get('rows')
    def identity(item: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        return item['source_index'], item['label'], item['parameter_hash']
    if not isinstance(rows, list) or not entries or len(rows) != len(entries):
        raise ValueError('prediction roster incomplete')
    expected = {identity(item): item for item in entries}
    predicted = {identity(item): item for item in rows}
    if len(expected) != len(entries) or len(predicted) != len(rows) or set(expected) != set(predicted):
        raise ValueError('prediction candidate identity differs or is duplicated')
    counts: dict[str, int] = {}
    nonfinite_spectrum_verifiable_count = 0
    for key, entry in expected.items():
        item = predicted[key]
        structure = FullCellStructureSpec.from_mapping(entry['structure'])
        if structure.parameter_hash != entry['parameter_hash']:
            raise ValueError('candidate parameter identity differs')
        grids = periodic_perturbation_grids(structure, resolution=protocol['raster_resolution'], offsets=tuple(offsets))
        grid_hashes = [hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest() for grid in grids]
        if item.get('grid_sha256') != grid_hashes or grid_hashes[offsets.index(0.0)] != entry['master_grid_sha256']:
            raise ValueError('prediction offset grid identity differs')
        status = item.get('status')
        counts[status] = counts.get(status, 0) + 1
        bands_raw = item.get('predicted_bands')
        if status == 'nonfinite_prediction':
            if bands_raw is not None or item.get('predicted_gap') is not None or item.get('predicted_margin') is not None:
                raise ValueError('nonfinite prediction must remain unscoreable')
            encoded = item.get('nonfinite_spectrum')
            if encoded is not None:
                spectrum = np.asarray(encoded, dtype=object)
                if spectrum.shape != (len(offsets), *checkpoint_output_shape):
                    raise ValueError('nonfinite spectrum shape differs')
                nonfinite = 0
                for value in spectrum.flat:
                    if isinstance(value, str) and value in {'nan', '+inf', '-inf'}:
                        nonfinite += 1
                    elif (isinstance(value, bool) or not isinstance(value, (int, float))
                          or not np.isfinite(value)):
                        raise ValueError('invalid nonfinite spectrum encoding')
                if not nonfinite:
                    raise ValueError('nonfinite spectrum contains no nonfinite value')
                nonfinite_spectrum_verifiable_count += 1
            continue
        if item.get('nonfinite_spectrum') is not None:
            raise ValueError('finite prediction cannot contain nonfinite spectrum')
        bands = np.asarray(bands_raw, dtype=np.float64)
        if bands.shape != (len(offsets), *checkpoint_output_shape) or not np.isfinite(bands).all():
            raise ValueError('prediction spectral shape or finiteness differs')
        unordered = bool(np.any(np.diff(bands, axis=-1) < 0))
        lower = manifest['lower_band']
        negative_pair = bool(np.any(bands[..., lower-1:lower+1] < 0))
        if status == 'band_order_violation':
            if not unordered or item.get('predicted_gap') is not None or item.get('predicted_margin') is not None:
                raise ValueError('band-order failure is inconsistent')
            continue
        if status == 'negative_frequency':
            if unordered or not negative_pair or item.get('predicted_gap') is not None or item.get('predicted_margin') is not None:
                raise ValueError('negative-frequency failure is inconsistent')
            continue
        if status != 'surrogate_only' or unordered or negative_pair:
            raise ValueError('invalid scoreable prediction')
        gap = asdict(common_gap(bands[..., lower-1:lower+1], lower_band=1,
                                target_frequency=manifest['target_frequency']))
        observed = item.get('predicted_gap')
        if (not isinstance(observed, Mapping) or set(observed) != set(gap)
                or observed['contains_target'] is not gap['contains_target']
                or any(isinstance(observed[k], bool) or not isinstance(observed[k], (int, float))
                       or not np.isclose(observed[k], gap[k], rtol=0, atol=1e-10)
                       for k in ('lower_edge', 'upper_edge', 'signed_width', 'target_margin'))
                or isinstance(item.get('predicted_margin'), bool)
                or not isinstance(item.get('predicted_margin'), (int, float))
                or not np.isclose(item['predicted_margin'], gap['target_margin'], rtol=0, atol=1e-10)):
            raise ValueError('prediction gap does not reproduce from spectrum')
    return {'scope': 'frozen_proxy_input_and_score_contract_only', 'candidate_count': len(entries),
            'status_counts': counts, 'nonfinite_failure_reason_verifiable': False,
            'nonfinite_spectrum_verifiable_count': nonfinite_spectrum_verifiable_count}


def validate_paired_proxy(proxy: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *,
                          manifest_sha256: str, protocol_sha256: str,
                          checkpoint_sha256: str, k_points_sha256: str,
                          allow_unscored: bool = False) -> list[dict[str, Any]]:
    """Check paired identities; optionally retain explicit invalid predictions unscored."""
    if (proxy.get('status') != 'exploratory_surrogate_common_gap_only'
            or proxy.get('manifest_sha256') != manifest_sha256 or proxy.get('protocol_sha256') != protocol_sha256
            or proxy.get('checkpoint_sha256') != checkpoint_sha256 or proxy.get('k_points_sha256') != k_points_sha256):
        raise ValueError('proxy provenance mismatch')
    def key(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        return row['source_index'], row['label'], row['parameter_hash']
    indexed = {key(item): item for item in records}
    if len(indexed) != len(records) or len(proxy['rows']) != len(records):
        raise ValueError('incomplete or duplicate candidate records')
    paired = []
    seen = set()
    for item in proxy['rows']:
        identity = key(item)
        if identity in seen or identity not in indexed:
            raise ValueError('proxy candidate identity mismatch')
        seen.add(identity)
        record = indexed[identity]
        if record.get('manifest_sha256') != manifest_sha256 or record.get('protocol_sha256') != protocol_sha256:
            raise ValueError('physical provenance mismatch')
        observations = record['observations']
        if ([r['offset'] for r in observations] != list(proxy['offsets'])
                or [r['provenance']['evaluated_grid_sha256'] for r in observations] != item['grid_sha256']):
            raise ValueError('offset grid identity mismatch')
        scored = item.get('status') == 'surrogate_only' and item.get('predicted_margin') is not None
        unscored = (allow_unscored is True and item.get('status') in
                    {'band_order_violation', 'nonfinite_prediction', 'negative_frequency'}
                    and item.get('predicted_margin') is None and item.get('predicted_gap') is None)
        if not scored and not unscored:
            raise ValueError('proxy candidate has invalid score')
        paired.append(dict(item))
    return paired


def run(manifest_path: str | Path, *, manifest_sha256: str, protocol_path: str | Path,
        protocol_sha256: str, checkpoint: str | Path, checkpoint_sha256: str,
        device: str, output: str | Path, parity_reference_device: str | None = None,
        parity_absolute_tolerance: float | None = None) -> Path:
    """Predict all candidate offsets; caller verifies checkpoint raster contract."""
    manifest, digest = read_checked_json(manifest_path, manifest_sha256)
    protocol, protocol_digest = read_checked_json(protocol_path, protocol_sha256)
    predictor = FrozenBandAdapter(checkpoint, sha256=checkpoint_sha256, device=device)
    points = np.ascontiguousarray(manifest['k_points'], dtype=np.float32)
    if hashlib.sha256(points.tobytes()).hexdigest() != predictor.k_points_sha256:
        raise ValueError('checkpoint k-point identity mismatch')
    if len(predictor.input_shape) != 2 or predictor.input_shape[0] != predictor.input_shape[1]:
        raise ValueError('square 2D predictor inputs required')
    offsets = tuple(protocol['offsets'])
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    if (parity_reference_device is None) != (parity_absolute_tolerance is None):
        raise ValueError('reference device and absolute tolerance must be supplied together')
    parity = None
    if parity_reference_device is not None:
        if not manifest['rows']:
            raise ValueError('device parity requires a candidate')
        first = manifest['rows'][0]
        structure = FullCellStructureSpec.from_mapping(first['structure'])
        if structure.parameter_hash != first['parameter_hash']:
            raise ValueError('parity candidate parameter identity mismatch')
        grids = periodic_perturbation_grids(structure, resolution=protocol['raster_resolution'], offsets=offsets)
        reference = FrozenBandAdapter(checkpoint, sha256=checkpoint_sha256, device=parity_reference_device)
        parity = compare_predictor_devices(grids, predictor, reference, input_resolution=predictor.input_shape[0],
                                           absolute_tolerance=parity_absolute_tolerance)
        parity.update(candidate_index=0, reference_device=parity_reference_device, device=device)
        if not parity['passed']:
            raise ValueError(f'device prediction parity failed: {parity}')
    started = time.monotonic()
    rows = []
    for source in manifest['rows']:
        structure = FullCellStructureSpec.from_mapping(source['structure'])
        if structure.parameter_hash != source['parameter_hash']:
            raise ValueError('parameter identity mismatch')
        grids = periodic_perturbation_grids(structure, resolution=protocol['raster_resolution'], offsets=offsets)
        result = predict_perturbed_gap(grids, predictor, input_resolution=predictor.input_shape[0],
                                       lower_band=manifest['lower_band'], target_frequency=manifest['target_frequency'])
        zero = offsets.index(0.0)
        if result['grid_sha256'][zero] != source['master_grid_sha256']:
            raise ValueError('master raster identity mismatch')
        row = {'source_index': source['source_index'], 'label': source['label'],
               'parameter_hash': structure.parameter_hash, 'grid_sha256': result['grid_sha256'],
               'status': result['status'], 'predicted_margin': None, 'predicted_gap': result['gap'],
               'predicted_bands': result.get('bands')}
        if 'nonfinite_spectrum' in result:
            row['nonfinite_spectrum'] = result['nonfinite_spectrum']
        if result['gap'] is not None:
            row['predicted_margin'] = result['gap']['target_margin']
        rows.append(row)
    payload = {'status': 'exploratory_surrogate_common_gap_only', 'manifest_sha256': digest,
               'protocol_sha256': protocol_digest, 'checkpoint_sha256': checkpoint_sha256,
               'k_points_sha256': predictor.k_points_sha256, 'offsets': offsets,
               'elapsed_s': time.monotonic() - started, 'rows': rows}
    if parity is not None:
        payload['device_parity'] = parity
    with output_path.open('x', encoding='utf8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Predict frozen candidate common gaps')
    for name in ('manifest', 'manifest-sha256', 'protocol', 'protocol-sha256',
                 'checkpoint', 'checkpoint-sha256', 'device', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--parity-reference-device')
    parser.add_argument('--parity-absolute-tolerance', type=float)
    args = parser.parse_args()
    print(run(args.manifest, manifest_sha256=args.manifest_sha256, protocol_path=args.protocol,
              protocol_sha256=args.protocol_sha256, checkpoint=args.checkpoint,
              checkpoint_sha256=args.checkpoint_sha256, device=args.device, output=args.output,
              parity_reference_device=args.parity_reference_device,
              parity_absolute_tolerance=args.parity_absolute_tolerance))


if __name__ == '__main__':
    main()
