"""Replay frozen forward checkpoints on their validation split.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from phc2d.forward.full_cell_evaluation import evaluate_full_cell
from phc2d.inverse.frozen_band_adapter import FrozenBandAdapter


def replay(*, dataset: str | Path, dataset_sha256: str, grid_field: str,
           checkpoint: str | Path, checkpoint_sha256: str, expected_model: str,
           tolerance_pct_points: float, batch_size: int, device: str, output: str | Path) -> Path:
    """Verify checkpoint reconstruction and validation metric without reading test targets."""
    import h5py
    import torch

    if not math.isfinite(tolerance_pct_points) or tolerance_pct_points < 0 or batch_size < 1:
        raise ValueError('invalid replay tolerance or batch size')
    dataset_path = Path(dataset)
    digest = hashlib.sha256()
    with dataset_path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != dataset_sha256:
        raise ValueError('dataset SHA-256 mismatch')
    predictor = FrozenBandAdapter(checkpoint, sha256=checkpoint_sha256, device=device)
    if predictor.model_name != expected_model or predictor.selection_split != 'validation':
        raise ValueError('model identity or selection split mismatch')
    reference = predictor.validation_bands_2_10_mre_pct
    if reference is None or not math.isfinite(float(reference)):
        raise ValueError('checkpoint validation reference missing')
    with h5py.File(dataset_path, 'r') as handle:
        required = {grid_field, 'Wh', 'k_points', 'idx_val', 'idx_test'}
        if not required.issubset(handle.keys()):
            raise ValueError('dataset fields missing')
        indices = np.asarray(handle['idx_val'],dtype=np.int64)
        test_indices = np.asarray(handle['idx_test'],dtype=np.int64)
        if (indices.ndim != 1 or test_indices.ndim != 1 or not len(indices)
                or len(np.unique(indices)) != len(indices)
                or len(np.unique(test_indices)) != len(test_indices)
                or np.any(indices < 0) or np.any(test_indices < 0)
                or np.any(indices >= len(handle[grid_field]))
                or np.any(test_indices >= len(handle[grid_field]))
                or np.intersect1d(indices,test_indices).size):
            raise ValueError('validation split identity invalid')
        points = np.ascontiguousarray(handle['k_points'],dtype=np.float32)
        if hashlib.sha256(points.tobytes()).hexdigest() != predictor.k_points_sha256:
            raise ValueError('dataset and checkpoint k-point identity mismatch')
        indices = np.sort(indices)
        grids = np.asarray(handle[grid_field][indices],dtype=np.float32)
        targets = np.asarray(handle['Wh'][indices],dtype=np.float32)
    if (tuple(grids.shape[1:]) != predictor.input_shape or tuple(targets.shape[1:]) != predictor.output_shape
            or not np.isfinite(grids).all() or not np.isfinite(targets).all()):
        raise ValueError('validation arrays violate checkpoint contract')
    mean = predictor.mean.detach().cpu().numpy()
    std = predictor.std.detach().cpu().numpy()
    normalized = (targets - mean) / std
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(grids),
        torch.from_numpy(normalized)),batch_size=batch_size,shuffle=False)
    def forward(_model, inputs, **_kwargs):
        return predictor.predict_normalized_tensor(inputs)
    started=time.monotonic()
    metrics=evaluate_full_cell(predictor.model,loader,device=predictor.device,mean=predictor.mean,
        std=predictor.std,model_spec=predictor.spec,k_points=predictor.k_points,lane_index=predictor.lane_index,
        torch_module=torch,forward_model=forward)
    difference=abs(float(metrics['bands_2_10_mre_pct'])-float(reference))
    payload={'status':'validation_replay_pass' if difference<=tolerance_pct_points else 'validation_replay_failed',
             'scope':'validation_only_no_test_targets','model':predictor.model_name,'checkpoint_sha256':checkpoint_sha256,
             'dataset_sha256':dataset_sha256,'grid_field':grid_field,'k_points_sha256':predictor.k_points_sha256,
             'validation_indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(), 'samples':len(indices),
             'recorded_validation_bands_2_10_mre_pct':float(reference),'recomputed_bands_2_10_mre_pct':metrics['bands_2_10_mre_pct'],
             'absolute_difference_pct_points':difference,'tolerance_pct_points':tolerance_pct_points,
             'elapsed_s':time.monotonic()-started}
    destination=Path(output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf8') as handle:
        json.dump(payload,handle,indent=2,allow_nan=False)
    return destination


def main() -> None:
    parser=argparse.ArgumentParser(description='Replay one checkpoint on validation data')
    for name in ('dataset','dataset-sha256','grid-field','checkpoint','checkpoint-sha256',
                 'expected-model','device','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--tolerance-pct-points',required=True,type=float)
    parser.add_argument('--batch-size',required=True,type=int)
    path = replay(**vars(parser.parse_args()))
    print(path)
    if json.loads(path.read_text(encoding='utf8'))['status'] != 'validation_replay_pass':
        raise SystemExit(1)


if __name__=='__main__':
    main()
