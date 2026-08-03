"""Shared inverse-design configuration dataclasses.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from phc2d.training import DeviceConfig


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _first_string(value: Any, *, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item:
                return item
    return default


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))


@dataclass(frozen=True)
class DispersionInverseConfig:
    """Configuration for dispersion inverse design with a frozen surrogate."""

    target_path: Path
    surrogate_kind: str
    checkpoint_path: Path
    output_dir: Path
    steps: int = 500
    seed: int = 0


@dataclass(frozen=True)
class GenerativeInverseConfig:
    """Configuration for candidate-based generative inverse screening."""

    source_path: Path = Path("")
    raw: dict[str, Any] = field(default_factory=dict)
    experiment_id: str = "generative_inverse"
    stage: str = "inverse_sample"
    condition_target: str = "Wh"
    target_split: str = "test"
    target_limit: int | None = None
    target_selection_seed: int | None = None
    target_manifest_path: Path | None = None

    dataset_path: Path = Path("data/tiny.npz")
    output_dir: Path = Path("runs")

    run_dir: Path | None = None
    model_name: str = "conditional_vae_rho"
    output_repr: str = "rho_wedge"
    seeds: tuple[int, ...] = (0,)
    train_epochs: int = 10
    checkpoint_epochs: tuple[int, ...] = field(default_factory=tuple)
    train_batch_size: int = 32
    learning_rate: float = 1.0e-3
    learning_rates: tuple[float, ...] = field(default_factory=tuple)
    optimizer: str = "adam"
    weight_decay: float = 0.0
    latent_dim: int = 8
    hidden_dim: int = 128
    depth: int = 2
    mode_count: int = 1
    generator_checkpoint_path: Path | None = None
    n_candidates: int = 16
    surrogate_top_m: int = 4
    seed: int = 0
    candidate_value_range: tuple[float, float] = (0.0, 1.0)
    threshold: float = 0.5
    morphology_filters: dict[str, float] = field(default_factory=dict)
    candidate_separation: dict[str, Any] = field(default_factory=dict)
    ranking: dict[str, float] = field(default_factory=dict)
    replay_enabled: bool = False
    allow_mock_generator: bool = True
    allow_mock_surrogate: bool = True
    surrogate: dict[str, Any] = field(default_factory=dict)
    rerank: dict[str, Any] = field(default_factory=dict)
    candidate_seed_offset: int = 0
    surrogate_seed_offset: int = 0
    replay_seed_offset: int = 0
    runtime: DeviceConfig = field(default_factory=DeviceConfig)
    metrics_include: tuple[str, ...] = (
        "validity_rate",
        "unique_rate",
        "diversity_hamming",
        "fill_fraction",
        "mean_morphology_score",
        "best_of_k_surrogate_mre",
        "candidate_sampling_time_s",
        "surrogate_ranking_time_s",
        "inference_time_per_sample_ms",
        "wall_time_total_s",
    )

    def __post_init__(self) -> None:
        if self.n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        if self.surrogate_top_m <= 0:
            raise ValueError("surrogate_top_m must be positive")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if self.train_epochs <= 0:
            raise ValueError("train_epochs must be positive")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if any(float(item) <= 0 for item in self.learning_rates):
            raise ValueError("learning_rates must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.mode_count <= 0:
            raise ValueError("mode_count must be positive")
        lo, hi = self.candidate_value_range
        if lo >= hi:
            raise ValueError("candidate_value_range must be an increasing pair")
        if self.morphology_filters is None:
            raise ValueError("morphology_filters must not be None")
        if self.candidate_separation is None:
            raise ValueError("candidate_separation must not be None")
        if self.ranking is None:
            raise ValueError("ranking must not be None")
        if self.output_repr not in {"rho_wedge", "rho_grid"}:
            raise ValueError("output_repr must be one of: rho_wedge, rho_grid")
        if self.target_limit is not None and self.target_limit <= 0:
            raise ValueError("target_limit must be positive when set")

    @property
    def runs_dir(self) -> Path:
        return self.output_dir

    def to_resolved_dict(self) -> dict[str, Any]:
        return {
            "experiment": {
                "experiment_id": self.experiment_id,
                "stage": self.stage,
            },
            "dataset": {
                "path": str(self.dataset_path),
                "condition_target": self.condition_target,
                "output_repr": self.output_repr,
                "target_split": self.target_split,
                "target_limit": self.target_limit,
                "target_selection_seed": self.target_selection_seed,
                "target_manifest_path": str(self.target_manifest_path)
                if self.target_manifest_path
                else None,
            },
            "model": {
                "name": self.model_name,
            },
            "inverse": {
                "n_candidates": self.n_candidates,
                "surrogate_top_m": self.surrogate_top_m,
                "seed": self.seed,
                "seeds": list(self.seeds),
                "train_epochs": self.train_epochs,
                "checkpoint_epochs": list(self.checkpoint_epochs),
                "train_batch_size": self.train_batch_size,
                "learning_rate": self.learning_rate,
                "learning_rates": list(self.learning_rates),
                "optimizer": self.optimizer,
                "weight_decay": self.weight_decay,
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "depth": self.depth,
                "mode_count": self.mode_count,
                "generator_checkpoint_path": str(self.generator_checkpoint_path)
                if self.generator_checkpoint_path
                else None,
                "candidate_seed_offset": self.candidate_seed_offset,
                "surrogate_seed_offset": self.surrogate_seed_offset,
                "replay_seed_offset": self.replay_seed_offset,
                "candidate_value_range": list(self.candidate_value_range),
                "threshold": self.threshold,
                "morphology_filters": dict(self.morphology_filters),
                "candidate_separation": dict(self.candidate_separation),
                "ranking": dict(self.ranking),
                "allow_mock_generator": self.allow_mock_generator,
                "allow_mock_surrogate": self.allow_mock_surrogate,
            },
            "surrogate": dict(self.surrogate),
            "rerank": dict(self.rerank),
            "runtime": self.runtime.to_dict(),
            "output": {
                "runs_dir": str(self.output_dir),
                "run_dir": str(self.run_dir) if self.run_dir else None,
            },
            "replay": {"enabled": self.replay_enabled},
        }

    def to_resolved_yaml(self) -> str:
        return yaml.safe_dump(self.to_resolved_dict(), sort_keys=False, allow_unicode=True)


def load_inverse_run_config(path: str | Path) -> GenerativeInverseConfig:
    """Load a generative inverse YAML config and resolve defaults."""

    source_path = Path(path)
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    raw = _as_mapping(raw, name="config")
    experiment_cfg = _as_mapping(raw.get("experiment"), name="experiment")
    dataset_cfg = _as_mapping(raw.get("dataset"), name="dataset")
    model_cfg = _as_mapping(raw.get("model"), name="model")
    inverse_cfg = _as_mapping(raw.get("inverse"), name="inverse")
    surrogate_cfg = _as_mapping(raw.get("surrogate"), name="surrogate")
    replay_cfg = _as_mapping(raw.get("replay"), name="replay")
    rerank_cfg = _as_mapping(raw.get("rerank"), name="rerank")
    output_cfg = _as_mapping(raw.get("output"), name="output")
    metrics_cfg = _as_mapping(raw.get("metrics"), name="metrics")
    runtime_cfg = _as_mapping(raw.get("runtime"), name="runtime")

    candidate_value_range = inverse_cfg.get("candidate_value_range", (0.0, 1.0))
    if len(candidate_value_range) != 2:
        raise ValueError("candidate_value_range must contain exactly two numeric bounds")

    experiment_id = str(
        experiment_cfg.get(
            "experiment_id",
            f"inverse_{_first_string(model_cfg.get('name'), default='cvae')}",
        )
    )

    runs_root = Path(str(output_cfg.get("runs_dir", output_cfg.get("run_dir", "runs"))))
    run_dir = _optional_path(output_cfg.get("run_dir"))
    metrics_include = tuple(_as_mapping(metrics_cfg, name="metrics").get("include", ()))
    if not metrics_include:
        metrics_include = (
            "validity_rate",
            "unique_rate",
            "diversity_hamming",
            "fill_fraction",
            "mean_morphology_score",
            "best_of_k_surrogate_mre",
            "candidate_sampling_time_s",
            "surrogate_ranking_time_s",
            "inference_time_per_sample_ms",
            "wall_time_total_s",
        )

    seed = int(model_cfg.get("seed", inverse_cfg.get("seed", 0)))
    raw_seeds = model_cfg.get("seeds", inverse_cfg.get("seeds", [seed]))
    if not isinstance(raw_seeds, list):
        raw_seeds = [raw_seeds]
    raw_learning_rates = inverse_cfg.get("learning_rates")
    if raw_learning_rates is None:
        raw_learning_rates = [inverse_cfg.get("learning_rate", inverse_cfg.get("lr", 1.0e-3))]
    if not isinstance(raw_learning_rates, list):
        raw_learning_rates = [raw_learning_rates]
    raw_checkpoint_epochs = inverse_cfg.get("checkpoint_epochs", ())
    if raw_checkpoint_epochs is None:
        checkpoint_epochs: tuple[int, ...] = ()
    elif isinstance(raw_checkpoint_epochs, (list, tuple)):
        checkpoint_epochs = tuple(sorted({int(item) for item in raw_checkpoint_epochs if int(item) > 0}))
    else:
        checkpoint_epochs = (int(raw_checkpoint_epochs),)

    return GenerativeInverseConfig(
        source_path=source_path,
        raw=dict(raw),
        experiment_id=experiment_id,
        stage=str(experiment_cfg.get("stage", "inverse_sample")),
        condition_target=_first_string(dataset_cfg.get("condition_target"), default="Wh"),
        target_split=_first_string(dataset_cfg.get("target_split", dataset_cfg.get("split")), default="test"),
        target_limit=(
            int(dataset_cfg.get("target_limit", inverse_cfg.get("target_limit")))
            if dataset_cfg.get("target_limit", inverse_cfg.get("target_limit")) is not None
            else None
        ),
        target_selection_seed=(
            int(dataset_cfg.get("target_selection_seed", inverse_cfg.get("target_selection_seed")))
            if dataset_cfg.get("target_selection_seed", inverse_cfg.get("target_selection_seed")) is not None
            else None
        ),
        target_manifest_path=_optional_path(dataset_cfg.get("target_manifest_path")),
        dataset_path=Path(str(dataset_cfg.get("path", "data/tiny.npz"))),
        output_dir=runs_root,
        run_dir=run_dir,
        model_name=_first_string(model_cfg.get("name"), default="conditional_vae_rho"),
        output_repr=_first_string(dataset_cfg.get("output_repr"), default="rho_wedge"),
        seeds=tuple(int(item) for item in raw_seeds),
        train_epochs=int(inverse_cfg.get("train_epochs", inverse_cfg.get("epochs", 10))),
        checkpoint_epochs=checkpoint_epochs,
        train_batch_size=int(inverse_cfg.get("train_batch_size", inverse_cfg.get("batch_size", 32))),
        learning_rate=float(inverse_cfg.get("learning_rate", inverse_cfg.get("lr", 1.0e-3))),
        learning_rates=tuple(float(item) for item in raw_learning_rates),
        optimizer=str(inverse_cfg.get("optimizer", "adam")),
        weight_decay=float(inverse_cfg.get("weight_decay", 0.0)),
        latent_dim=int(model_cfg.get("latent_dim", inverse_cfg.get("latent_dim", 8))),
        hidden_dim=int(model_cfg.get("hidden_dim", inverse_cfg.get("hidden_dim", 128))),
        depth=int(model_cfg.get("depth", inverse_cfg.get("depth", 2))),
        mode_count=int(model_cfg.get("mode_count", inverse_cfg.get("mode_count", 1))),
        generator_checkpoint_path=_optional_path(
            inverse_cfg.get("generator_checkpoint_path", model_cfg.get("checkpoint_path"))
        ),
        n_candidates=int(inverse_cfg.get("n_candidates", 16)),
        surrogate_top_m=int(inverse_cfg.get("surrogate_top_m", 4)),
        seed=seed,
        candidate_value_range=(float(candidate_value_range[0]), float(candidate_value_range[1])),
        threshold=float(inverse_cfg.get("threshold", 0.5)),
        morphology_filters=_as_mapping(inverse_cfg.get("morphology_filters"), name="inverse.morphology_filters"),
        candidate_separation=_as_mapping(
            inverse_cfg.get("candidate_separation"),
            name="inverse.candidate_separation",
        ),
        ranking=_as_mapping(inverse_cfg.get("ranking"), name="inverse.ranking"),
        replay_enabled=_coerce_bool(replay_cfg.get("enabled"), default=False),
        allow_mock_generator=_coerce_bool(inverse_cfg.get("allow_mock_generator"), default=True),
        allow_mock_surrogate=_coerce_bool(inverse_cfg.get("allow_mock_surrogate"), default=True),
        surrogate=surrogate_cfg,
        rerank=rerank_cfg,
        candidate_seed_offset=int(inverse_cfg.get("candidate_seed_offset", 0)),
        surrogate_seed_offset=int(inverse_cfg.get("surrogate_seed_offset", 0)),
        replay_seed_offset=int(inverse_cfg.get("replay_seed_offset", 0)),
        runtime=DeviceConfig.from_mapping(runtime_cfg),
        metrics_include=tuple(metrics_include),
    )


__all__ = ["DispersionInverseConfig", "GenerativeInverseConfig", "load_inverse_run_config"]
