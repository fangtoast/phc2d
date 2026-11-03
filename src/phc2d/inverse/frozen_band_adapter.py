"""Frozen full-cell band predictions in the checkpoint's physical units.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
import json
import os
from pathlib import Path
from typing import Any


def audit_singleton_inference_evidence(directory: str | Path, *, max_inferences: int) -> dict[str, Any]:
    """Inspect saved starts and outputs; an unfinished or orphan record remains unknown.

    This validates internal evidence only. The caller authenticates its origin,
    retains immutable copies and establishes that the original process stopped.
    """
    import numpy as np

    if isinstance(max_inferences, bool) or not isinstance(max_inferences, int) or max_inferences < 1:
        raise ValueError('positive integer inference budget required')
    root = Path(directory)
    requests = sorted(root.glob('request_*.json'))
    known_names, known_inputs = set(), set()
    starts, completed, hits, denials, unknown = 0, 0, 0, 0, []
    for number, path in enumerate(requests, 1):
        if path.name != f'request_{number:04d}.json' or path.is_symlink():
            raise ValueError('inference request sequence mismatch')
        known_names.add(path.name)
        event = json.loads(path.read_bytes())
        kind = event.get('kind')
        if kind == 'inference_started':
            starts += 1
            if (starts > max_inferences or event.get('inference') != starts
                    or event.get('input_file') != f'input_{starts:04d}.npy'):
                raise ValueError('inference identity or budget mismatch')
            input_path = root/event['input_file']
            raw = input_path.read_bytes()
            if input_path.is_symlink() or hashlib.sha256(raw).hexdigest() != event['input_file_sha256']:
                raise ValueError('inference input file mismatch')
            raster = np.load(input_path, allow_pickle=False)
            if (raster.dtype != np.float32 or raster.ndim != 3 or raster.shape[0] != 1
                    or hashlib.sha256(np.ascontiguousarray(raster).tobytes()).hexdigest() != event['input_sha256']):
                raise ValueError('inference input identity mismatch')
            known_names.add(input_path.name)
            completion = root/f'completed_{starts:04d}.json'
            if not completion.is_file():
                unknown.append(starts)
                continue
            data = json.loads(completion.read_bytes())
            output = root/f'output_{starts:04d}.npy'
            if (completion.is_symlink() or output.is_symlink() or data.get('inference') != starts
                    or data.get('output_file') != output.name
                    or hashlib.sha256(output.read_bytes()).hexdigest() != data.get('output_file_sha256')):
                raise ValueError('inference output file mismatch')
            np.load(output, allow_pickle=False)
            known_names.update((completion.name, output.name))
            known_inputs.add(event['input_sha256'])
            completed += 1
        elif kind == 'cache_hit':
            if event.get('input_sha256') not in known_inputs:
                raise ValueError('cache hit has no completed input')
            hits += 1
        elif kind == 'budget_exhausted':
            if starts < max_inferences:
                raise ValueError('premature budget denial')
            denials += 1
        else:
            raise ValueError('invalid inference event')
    extras = sorted(path.name for path in root.iterdir() if path.name not in known_names)
    return {'status': 'unknown' if unknown or extras else 'complete', 'requests': len(requests),
            'started': starts, 'completed': completed, 'cache_hits': hits, 'budget_denials': denials,
            'unknown_inferences': unknown, 'unmatched_files': extras}


def audit_completed_inference_cost(directory: str | Path, cost: dict[str, Any]) -> dict[str, Any]:
    """Reconcile a successful cost record against the individual saved calls."""
    if cost.get('error') is not None or cost.get('archive_error') is not None:
        raise ValueError('successful complete cost record required')
    audit = audit_singleton_inference_evidence(directory, max_inferences=cost['max_inferences'])
    if (audit['status'] != 'complete' or audit['budget_denials'] != 0
            or cost['requests'] != audit['requests'] or cost['inferences'] != audit['started']
            or cost['successful_inferences'] != audit['completed'] or cost['cache_hits'] != audit['cache_hits']):
        raise ValueError('cost record and inference evidence differ')
    return {**audit, 'status': 'cost_consistent_not_source_authentication'}


def audit_prediction_archive_against_calls(directory: str | Path, archive_path: str | Path,
                                           cost: dict[str, Any]) -> dict[str, Any]:
    """Compare archived tensors bitwise with the separately persisted calls."""
    import numpy as np

    audit = audit_completed_inference_cost(directory, cost)
    root = Path(directory)
    expected = {}
    for index in range(1, audit['completed'] + 1):
        x = np.load(root/f'input_{index:04d}.npy', allow_pickle=False)
        y = np.load(root/f'output_{index:04d}.npy', allow_pickle=False)
        key = (x.dtype.str, x.shape, x.tobytes())
        if key in expected:
            raise ValueError('duplicate actual input evidence')
        expected[key] = (y.dtype.str, y.shape, y.tobytes())
    with np.load(archive_path, allow_pickle=False) as archive:
        names = {f'{kind}_{index:04d}' for index in range(len(expected)) for kind in ('input', 'output')}
        if set(archive.files) != names or len(archive.files) != len(names):
            raise ValueError('prediction archive roster differs from calls')
        observed = {}
        for index in range(len(expected)):
            x, y = archive[f'input_{index:04d}'], archive[f'output_{index:04d}']
            key = (x.dtype.str, x.shape, x.tobytes())
            if key in observed:
                raise ValueError('duplicate archived input')
            observed[key] = (y.dtype.str, y.shape, y.tobytes())
    if observed != expected:
        raise ValueError('prediction archive and call evidence differ')
    return {'status': 'archive_matches_calls_not_source_authentication', 'unique_inferences': len(expected)}


class SingleSampleBudgetedPredictor:
    """Count and cap singleton inferences, reusing identical inputs from a bounded cache.

    Use only with a frozen deterministic predictor and a fixed input contract.
    Failed calls consume budget; the caller persists counts and any failure.
    This wrapper does not account for calls made directly to the wrapped object.
    """

    def __init__(self, predictor: Any, *, max_inferences: int, cache_capacity: int = 1,
                 evidence_directory: str | Path | None = None) -> None:
        if isinstance(max_inferences, bool) or not isinstance(max_inferences, int) or max_inferences < 1:
            raise ValueError('positive integer inference budget required')
        if (isinstance(cache_capacity, bool) or not isinstance(cache_capacity, int)
                or not 1 <= cache_capacity <= max_inferences):
            raise ValueError('cache capacity must fit the inference budget')
        self.predictor = predictor
        self.max_inferences = max_inferences
        self.requests = 0
        self.inferences = 0
        self.successful_inferences = 0
        self.cache_hits = 0
        self.cache_capacity = cache_capacity
        self._cache = OrderedDict()
        self.evidence_directory = None if evidence_directory is None else Path(evidence_directory)
        if self.evidence_directory is not None:
            self.evidence_directory.mkdir(parents=False, exist_ok=False)

    def _save_array(self, name: str, tensor: Any) -> str:
        import numpy as np

        path = self.evidence_directory/name
        with path.open('xb') as stream:
            np.save(stream, tensor.detach().cpu().contiguous().numpy(), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _save_event(self, name: str, event: dict[str, Any]) -> None:
        with (self.evidence_directory/name).open('x', encoding='utf8') as stream:
            json.dump(event, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())

    def predict_tensor(self, inputs: Any) -> Any:
        import torch

        if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 3 or inputs.shape[0] != 1
                or inputs.dtype != torch.float32 or inputs.requires_grad):
            raise ValueError('one non-gradient float32 raster required')
        array = inputs.detach().cpu().contiguous()
        key = (tuple(array.shape), array.numpy().tobytes())
        self.requests += 1
        if key in self._cache:
            if self.evidence_directory is not None:
                self._save_event(f'request_{self.requests:04d}.json',
                    {'kind': 'cache_hit', 'input_sha256': hashlib.sha256(key[1]).hexdigest()})
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key].clone()
        if self.inferences >= self.max_inferences:
            if self.evidence_directory is not None:
                self._save_event(f'request_{self.requests:04d}.json',
                    {'kind': 'budget_exhausted', 'input_sha256': hashlib.sha256(key[1]).hexdigest()})
            raise ValueError('surrogate inference budget exhausted')
        self.inferences += 1
        if self.evidence_directory is not None:
            input_name = f'input_{self.inferences:04d}.npy'
            input_file_sha = self._save_array(input_name, array)
            self._save_event(f'request_{self.requests:04d}.json',
                {'kind': 'inference_started', 'inference': self.inferences,
                 'input_file': input_name, 'input_file_sha256': input_file_sha,
                 'input_sha256': hashlib.sha256(key[1]).hexdigest()})
        output = self.predictor.predict_tensor(inputs)
        if not isinstance(output, torch.Tensor):
            raise ValueError('predictor must return a tensor')
        if self.evidence_directory is not None:
            output_name = f'output_{self.inferences:04d}.npy'
            output_file_sha = self._save_array(output_name, output)
            self._save_event(f'completed_{self.inferences:04d}.json',
                {'inference': self.inferences, 'output_file': output_name, 'output_file_sha256': output_file_sha})
        self.successful_inferences += 1
        self._cache[key] = output.detach().clone()
        if len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)
        return output

    def save_prediction_archive(self, path: str | Path) -> None:
        """Save complete input/output tensors without JSON float conversion.

        A cache covering the budget is required so no successful output can be
        silently omitted. The caller binds this artifact to the trajectory and
        independently audits identities and completeness.
        """
        import numpy as np

        if self.cache_capacity != self.max_inferences or len(self._cache) != self.successful_inferences:
            raise ValueError('complete prediction cache required')
        arrays = {}
        for index, (key, output) in enumerate(self._cache.items()):
            shape, raw = key
            arrays[f'input_{index:04d}'] = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
            arrays[f'output_{index:04d}'] = output.cpu().contiguous().numpy()
        with Path(path).open('xb') as stream:
            np.savez(stream, **arrays)


class FrozenBandAdapter:
    """Load a verified single-lane forward checkpoint without altering its identity.

    Inputs must already obey the checkpoint's raster and sampling contract.
    The caller must independently verify that contract against the source dataset.
    """

    def __init__(self, checkpoint: str | Path, *, sha256: str, device: str = "cpu") -> None:
        import torch

        from phc2d.models import build_model, get_model_spec

        path = Path(checkpoint)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != sha256.lower():
            raise ValueError("checkpoint SHA-256 mismatch")
        payload = torch.load(path, map_location=device, weights_only=True)
        if "target_mean" not in payload or "target_std" not in payload:
            raise ValueError("single-lane checkpoint normalizer required")
        self.torch = torch
        self.device = torch.device(device)
        self.input_shape = tuple(int(v) for v in payload["input_shape"])
        self.output_shape = tuple(int(v) for v in payload["output_shape"])
        self.mean = torch.as_tensor(payload["target_mean"], device=self.device)
        self.std = torch.as_tensor(payload["target_std"], device=self.device)
        if tuple(self.mean.shape) != self.output_shape or tuple(self.std.shape) != self.output_shape:
            raise ValueError("normalizer shape does not match checkpoint output")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("normalizer must be finite with positive standard deviation")
        self.spec = get_model_spec(str(payload["model_name"]))
        self.k_points = payload.get("k_points")
        if self.k_points is not None:
            points = torch.as_tensor(self.k_points, dtype=torch.float32, device="cpu").contiguous()
            if points.ndim != 2 or not torch.isfinite(points).all():
                raise ValueError("checkpoint k points must be a finite coordinate matrix")
            expected_k_hash = payload.get("k_points_sha256")
            if expected_k_hash and hashlib.sha256(points.numpy().tobytes()).hexdigest() != expected_k_hash:
                raise ValueError("checkpoint k-point SHA-256 mismatch")
            if len(self.output_shape) != 2 or points.shape[0] != self.output_shape[0]:
                raise ValueError("checkpoint k points do not match output k axis")
            self.k_points = points.to(self.device)
        self.lane_index = payload.get("lane_index")
        kwargs = dict(payload["model_params"])
        if self.spec.accepts_k_points:
            if self.k_points is None:
                raise ValueError("checkpoint requires physical k points")
            kwargs["k_points"] = self.k_points
        if self.spec.accepts_lane_id and self.lane_index is None:
            raise ValueError("checkpoint requires lane identity")
        self.model = build_model(self.spec.name, input_shape=self.input_shape,
                                 output_shape=self.output_shape, **kwargs).to(self.device)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model_name = self.spec.name
        self.checkpoint_sha256 = digest.hexdigest()
        self.k_points_sha256 = payload.get("k_points_sha256")
        self.selection_split = payload.get("selection_split")
        self.validation_bands_2_10_mre_pct = payload.get("validation_bands_2_10_mre_pct")

    def predict_normalized_tensor(self, inputs: Any) -> Any:
        """Predict normalized checkpoint outputs and preserve input gradients."""
        torch = self.torch
        x = inputs.to(device=self.device, dtype=torch.float32)
        if tuple(x.shape[1:]) != self.input_shape or x.shape[0] == 0:
            raise ValueError("input shape does not match checkpoint")
        if not torch.isfinite(x).all():
            raise ValueError("inputs must be finite")
        kwargs: dict[str, Any] = {}
        if self.spec.accepts_k_points:
            kwargs["k_points"] = self.k_points
        if self.spec.accepts_lane_id:
            kwargs["lane_id"] = torch.full((x.shape[0],), int(self.lane_index),
                                           dtype=torch.long, device=self.device)
        return self.model(x, **kwargs).reshape(x.shape[0], *self.output_shape)

    def predict_tensor(self, inputs: Any) -> Any:
        """Predict frequencies while preserving gradients with respect to inputs."""
        return self.predict_normalized_tensor(inputs) * self.std + self.mean
