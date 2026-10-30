"""Frozen full-cell band predictions in the checkpoint's physical units.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


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
            self.k_points = torch.as_tensor(self.k_points, dtype=torch.float32, device=self.device)
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

    def predict_tensor(self, inputs: Any) -> Any:
        """Predict frequencies while preserving gradients with respect to inputs."""
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
        prediction = self.model(x, **kwargs).reshape(x.shape[0], *self.output_shape)
        return prediction * self.std + self.mean
