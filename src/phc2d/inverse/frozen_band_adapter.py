"""Frozen full-cell band predictions in the checkpoint's physical units.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any


class SingleSampleBudgetedPredictor:
    """Count and cap singleton inferences, reusing identical inputs from a bounded cache.

    Use only with a frozen deterministic predictor and a fixed input contract.
    Failed calls consume budget; the caller persists counts and any failure.
    This wrapper does not account for calls made directly to the wrapped object.
    """

    def __init__(self, predictor: Any, *, max_inferences: int, cache_capacity: int = 1) -> None:
        if isinstance(max_inferences, bool) or not isinstance(max_inferences, int) or max_inferences < 1:
            raise ValueError('positive integer inference budget required')
        if (isinstance(cache_capacity, bool) or not isinstance(cache_capacity, int)
                or not 1 <= cache_capacity <= max_inferences):
            raise ValueError('cache capacity must fit the inference budget')
        self.predictor = predictor
        self.max_inferences = max_inferences
        self.requests = 0
        self.inferences = 0
        self.cache_hits = 0
        self.cache_capacity = cache_capacity
        self._cache = OrderedDict()

    def predict_tensor(self, inputs: Any) -> Any:
        import torch

        if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 3 or inputs.shape[0] != 1
                or inputs.dtype != torch.float32 or inputs.requires_grad):
            raise ValueError('one non-gradient float32 raster required')
        array = inputs.detach().cpu().contiguous()
        key = (tuple(array.shape), array.numpy().tobytes())
        self.requests += 1
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key].clone()
        if self.inferences >= self.max_inferences:
            raise ValueError('surrogate inference budget exhausted')
        self.inferences += 1
        output = self.predictor.predict_tensor(inputs)
        if not isinstance(output, torch.Tensor):
            raise ValueError('predictor must return a tensor')
        self._cache[key] = output.detach().clone()
        if len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)
        return output


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
