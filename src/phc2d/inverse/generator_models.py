"""Neural generator architectures used by the inverse-design runtime.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any, Mapping

from phc2d.inverse.config import GenerativeInverseConfig


class _ConditionalGenerator:
    """Small conditional rho generator trained from ``Wh -> rho`` pairs."""

    def __init__(
        self,
        *,
        condition_dim: int,
        candidate_dim: int,
        latent_dim: int,
        hidden_dim: int,
        torch: Any,
    ) -> None:
        _, nn = torch, torch.nn
        self.condition_dim = int(condition_dim)
        self.candidate_dim = int(candidate_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.module = nn.Sequential(
            nn.Linear(self.condition_dim + self.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.candidate_dim),
            nn.Sigmoid(),
        )

    def to(self, device: Any) -> "_ConditionalGenerator":
        self.module.to(device)
        return self

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def parameters(self) -> Any:
        return self.module.parameters()

    def reconstruct(self, condition: Any, torch: Any) -> Any:
        noise = torch.zeros(
            (int(condition.shape[0]), self.latent_dim),
            dtype=condition.dtype,
            device=condition.device,
        )
        return self.module(torch.cat([condition, noise], dim=1))

    def sample(self, condition: Any, *, n_candidates: int, seed: int, torch: Any, device: Any) -> Any:
        generator = torch.Generator(device=str(device))
        generator.manual_seed(int(seed))
        cond = condition.reshape(1, -1).expand(int(n_candidates), -1)
        noise = torch.randn((int(n_candidates), self.latent_dim), generator=generator, device=device)
        return self.module(torch.cat([cond, noise], dim=1))

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.module.load_state_dict(dict(state_dict))


class _MixtureConditionalGenerator:
    """Mode-conditioned latent generator for one-to-many inverse candidates."""

    def __init__(
        self,
        *,
        condition_dim: int,
        candidate_dim: int,
        latent_dim: int,
        hidden_dim: int,
        depth: int,
        mode_count: int,
        torch: Any,
    ) -> None:
        nn = torch.nn
        self.condition_dim = int(condition_dim)
        self.candidate_dim = int(candidate_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.mode_count = int(mode_count)
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.mode_embedding = nn.Embedding(self.mode_count, self.hidden_dim)
        layers: list[Any] = []
        in_dim = self.hidden_dim * 2 + self.latent_dim
        for layer_idx in range(max(self.depth, 1)):
            layers.append(nn.Linear(in_dim if layer_idx == 0 else self.hidden_dim, self.hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(self.hidden_dim, self.candidate_dim))
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)

    def to(self, device: Any) -> "_MixtureConditionalGenerator":
        self.condition_encoder.to(device)
        self.mode_embedding.to(device)
        self.decoder.to(device)
        return self

    def train(self) -> None:
        self.condition_encoder.train()
        self.mode_embedding.train()
        self.decoder.train()

    def eval(self) -> None:
        self.condition_encoder.eval()
        self.mode_embedding.eval()
        self.decoder.eval()

    def parameters(self) -> Any:
        return list(self.condition_encoder.parameters()) + list(self.mode_embedding.parameters()) + list(self.decoder.parameters())

    def _decode(self, condition: Any, noise: Any, mode_ids: Any, torch: Any) -> Any:
        encoded = self.condition_encoder(condition)
        mode = self.mode_embedding(mode_ids)
        return self.decoder(torch.cat([encoded, mode, noise], dim=1))

    def reconstruct(self, condition: Any, torch: Any) -> Any:
        batch = int(condition.shape[0])
        noise = torch.zeros((batch, self.latent_dim), dtype=condition.dtype, device=condition.device)
        mode_ids = torch.arange(batch, device=condition.device, dtype=torch.long) % self.mode_count
        return self._decode(condition, noise, mode_ids, torch)

    def sample(self, condition: Any, *, n_candidates: int, seed: int, torch: Any, device: Any) -> Any:
        generator = torch.Generator(device=str(device))
        generator.manual_seed(int(seed))
        cond = condition.reshape(1, -1).expand(int(n_candidates), -1)
        noise = torch.randn((int(n_candidates), self.latent_dim), generator=generator, device=device)
        mode_ids = torch.arange(int(n_candidates), device=device, dtype=torch.long) % self.mode_count
        return self._decode(cond, noise, mode_ids, torch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "condition_encoder": self.condition_encoder.state_dict(),
            "mode_embedding": self.mode_embedding.state_dict(),
            "decoder": self.decoder.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if "condition_encoder" in state_dict:
            self.condition_encoder.load_state_dict(dict(state_dict["condition_encoder"]))
            self.mode_embedding.load_state_dict(dict(state_dict["mode_embedding"]))
            self.decoder.load_state_dict(dict(state_dict["decoder"]))
        else:
            raise ValueError("mixture_conditional_generator checkpoint is missing mixture state keys")


class _DirectInverseGenerator:
    """Deterministic direct ``Wh -> rho`` baseline used for ablation."""

    def __init__(
        self,
        *,
        condition_dim: int,
        candidate_dim: int,
        hidden_dim: int,
        torch: Any,
    ) -> None:
        nn = torch.nn
        self.condition_dim = int(condition_dim)
        self.candidate_dim = int(candidate_dim)
        self.latent_dim = 0
        self.hidden_dim = int(hidden_dim)
        self.module = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.candidate_dim),
            nn.Sigmoid(),
        )

    def to(self, device: Any) -> "_DirectInverseGenerator":
        self.module.to(device)
        return self

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def parameters(self) -> Any:
        return self.module.parameters()

    def reconstruct(self, condition: Any, torch: Any) -> Any:
        _ = torch
        return self.module(condition)

    def sample(self, condition: Any, *, n_candidates: int, seed: int, torch: Any, device: Any) -> Any:
        _ = seed, device
        base = self.module(condition.reshape(1, -1))
        return base.expand(int(n_candidates), -1).clone()

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.module.load_state_dict(dict(state_dict))


def _generator_kind(cfg: GenerativeInverseConfig, payload: Mapping[str, Any] | None = None) -> str:
    value = None if payload is None else payload.get("generator_kind")
    if value is None:
        value = cfg.model_name
    normalized = str(value).strip().lower()
    if normalized in {"direct_inverse_mlp", "conditional_mlp_inverse", "direct_mlp"}:
        return "direct_inverse_mlp"
    if normalized in {"mixture_conditional_generator", "mixture_conditional", "mixture"}:
        return "mixture_conditional_generator"
    return "conditional_latent_mlp"


def _make_generator_network(
    cfg: GenerativeInverseConfig,
    *,
    condition_dim: int,
    candidate_dim: int,
    torch: Any,
) -> Any:
    if _generator_kind(cfg) == "direct_inverse_mlp":
        return _DirectInverseGenerator(
            condition_dim=condition_dim,
            candidate_dim=candidate_dim,
            hidden_dim=int(cfg.hidden_dim),
            torch=torch,
        )
    if _generator_kind(cfg) == "mixture_conditional_generator":
        return _MixtureConditionalGenerator(
            condition_dim=condition_dim,
            candidate_dim=candidate_dim,
            latent_dim=int(cfg.latent_dim),
            hidden_dim=int(cfg.hidden_dim),
            depth=int(cfg.depth),
            mode_count=int(cfg.mode_count),
            torch=torch,
        )
    return _ConditionalGenerator(
        condition_dim=condition_dim,
        candidate_dim=candidate_dim,
        latent_dim=int(cfg.latent_dim),
        hidden_dim=int(cfg.hidden_dim),
        torch=torch,
    )
