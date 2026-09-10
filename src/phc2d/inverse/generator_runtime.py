"""Checkpoint and candidate adapters for inverse generator architectures.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from phc2d.data.lift import lift_rho_to_grid
from phc2d.inverse.candidates import CandidateBatch
from phc2d.inverse.config import GenerativeInverseConfig
from phc2d.inverse.generator_models import (
    _ConditionalGenerator,
    _DirectInverseGenerator,
    _MixtureConditionalGenerator,
    _generator_kind,
)
from phc2d.inverse.runtime_support import _to_jsonable


class _MockConditionalGenerator:
    """Dependency-light conditional VAE-like sampler for smoke tests."""

    def __init__(
        self,
        cfg: GenerativeInverseConfig,
        dim: int,
        target_seed_anchor: float,
        seed: int,
        *,
        torch: Any,
        device: Any,
    ) -> None:
        self.cfg = cfg
        self.dim = int(dim)
        self.seed = int(seed)
        self.target_seed_anchor = float(target_seed_anchor)
        self.torch = torch
        self.device = device

    def sample_tensor(self, target: Any, *, target_id: str, target_index: int) -> Any:
        _ = target
        generator = self.torch.Generator(device=str(self.device))
        generator.manual_seed(int(self.seed))
        base = self.torch.linspace(0.0, float(np.pi), self.dim, device=self.device, dtype=self.torch.float32)
        loc = 0.5 + 0.2 * self.torch.sin(base + float(self.target_seed_anchor))
        raw = self.torch.normal(
            mean=loc.expand(int(self.cfg.n_candidates), self.dim),
            std=0.15,
            generator=generator,
        )
        return self.torch.clamp(
            raw,
            min=float(self.cfg.candidate_value_range[0]),
            max=float(self.cfg.candidate_value_range[1]),
        )

    def to_candidate_batch(
        self,
        raw: Any,
        *,
        target: np.ndarray,
        target_id: str,
        target_index: int,
        metadata_indices: np.ndarray | None = None,
    ) -> CandidateBatch:
        raw_np = raw.detach().cpu().numpy().astype(np.float32)
        all_metadata = [
            {
                "candidate_index": int(i),
                "target_index": int(target_index),
                "sample_seed": int(self.seed),
                "target_id": str(target_id),
            }
            for i in range(self.cfg.n_candidates)
        ]
        if metadata_indices is None:
            metadata = all_metadata[: int(raw_np.shape[0])]
        else:
            metadata = [all_metadata[int(i)] for i in metadata_indices]
        if self.cfg.output_repr == "rho_grid":
            grid = np.stack([lift_rho_to_grid(row) for row in raw_np], axis=0)
            return CandidateBatch(
                target_id=target_id,
                condition={"target": _to_jsonable(target)},
                grid_candidates=grid,
                input_repr=self.cfg.output_repr,
                metadata=metadata,
            )
        return CandidateBatch(
            target_id=target_id,
            condition={"target": _to_jsonable(target)},
            rho_candidates=raw_np,
            input_repr=self.cfg.output_repr,
            metadata=metadata,
        )

class _TrainedConditionalGenerator:
    """Checkpoint-backed candidate sampler."""

    def __init__(
        self,
        cfg: GenerativeInverseConfig,
        checkpoint_path: Path,
        *,
        torch: Any,
        device: Any,
    ) -> None:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.cfg = cfg
        self.seed = int(payload.get("seed", cfg.seed))
        self.torch = torch
        self.device = device
        generator_kind = _generator_kind(cfg, payload)
        if generator_kind == "direct_inverse_mlp":
            self.net = _DirectInverseGenerator(
                condition_dim=int(payload["condition_dim"]),
                candidate_dim=int(payload["candidate_dim"]),
                hidden_dim=int(payload["hidden_dim"]),
                torch=torch,
            ).to(device)
        elif generator_kind == "mixture_conditional_generator":
            self.net = _MixtureConditionalGenerator(
                condition_dim=int(payload["condition_dim"]),
                candidate_dim=int(payload["candidate_dim"]),
                latent_dim=int(payload["latent_dim"]),
                hidden_dim=int(payload["hidden_dim"]),
                depth=int(payload.get("depth", cfg.depth)),
                mode_count=int(payload.get("mode_count", cfg.mode_count)),
                torch=torch,
            ).to(device)
        else:
            self.net = _ConditionalGenerator(
                condition_dim=int(payload["condition_dim"]),
                candidate_dim=int(payload["candidate_dim"]),
                latent_dim=int(payload["latent_dim"]),
                hidden_dim=int(payload["hidden_dim"]),
                torch=torch,
            ).to(device)
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()

    def sample_tensor(self, target: Any, *, target_id: str, target_index: int) -> Any:
        _ = target_id, target_index
        with self.torch.no_grad():
            return self.net.sample(
                target.reshape(-1),
                n_candidates=self.cfg.n_candidates,
                seed=self.seed,
                torch=self.torch,
                device=self.device,
            )

    def to_candidate_batch(
        self,
        raw: Any,
        *,
        target: np.ndarray,
        target_id: str,
        target_index: int,
        metadata_indices: np.ndarray | None = None,
    ) -> CandidateBatch:
        raw_np = raw.detach().cpu().numpy().astype(np.float32)
        all_metadata = [
            {
                "candidate_index": int(i),
                "target_index": int(target_index),
                "generator_seed": int(self.seed),
                "target_id": str(target_id),
                "generator_checkpoint": "trained",
            }
            for i in range(self.cfg.n_candidates)
        ]
        if metadata_indices is None:
            metadata = all_metadata
        else:
            metadata = [all_metadata[int(i)] for i in metadata_indices]
        if self.cfg.output_repr == "rho_grid":
            grid = np.stack([lift_rho_to_grid(row) for row in raw_np], axis=0)
            return CandidateBatch(
                target_id=target_id,
                condition={"target": _to_jsonable(target)},
                grid_candidates=grid,
                input_repr=self.cfg.output_repr,
                metadata=metadata,
            )
        return CandidateBatch(
            target_id=target_id,
            condition={"target": _to_jsonable(target)},
            rho_candidates=raw_np,
            input_repr=self.cfg.output_repr,
            metadata=metadata,
        )


# Stable public adapter names. The aliases intentionally preserve the historical
# class objects so existing imports and serialized identities remain unchanged.
MockConditionalGenerator = _MockConditionalGenerator
TrainedConditionalGenerator = _TrainedConditionalGenerator

__all__ = [
    "MockConditionalGenerator",
    "TrainedConditionalGenerator",
]
