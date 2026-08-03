"""Configuration loading for native forward train/eval runs.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from phc2d.data import DataRepresentationSpec, infer_dataset_schema
from phc2d.data.lift import infer_square_grid_shape_from_wedge_length
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


def _first_model_name(model_cfg: Mapping[str, Any]) -> str:
    if isinstance(model_cfg.get("name"), str):
        return str(model_cfg["name"])
    candidates = model_cfg.get("candidates")
    if isinstance(candidates, list):
        for name in candidates:
            if isinstance(name, str) and name != "pod_res_mlp":
                return name
        for name in candidates:
            if isinstance(name, str):
                return name
    return "res_mlp"


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


def _coerce_seed_list(value: Any, *, default: int) -> list[int]:
    if value is None:
        return [int(default)]
    if isinstance(value, bool):
        return [1 if value else 0]
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return [int(default)]
        try:
            return [int(text)]
        except ValueError as exc:
            raise ValueError(f"seeds must be integer or list/tuple of integers, got {value!r}") from exc
    if isinstance(value, (list, tuple, set)):
        values = [int(v) for v in value]
        if not values:
            return [int(default)]
        return values
    raise ValueError(f"seeds must be integer or list/tuple of integers, got {value!r}")


def _tuple2(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    seq = list(value)
    if len(seq) != 2:
        raise ValueError(f"Expected a 2D grid_shape, got {value!r}")
    return (int(seq[0]), int(seq[1]))


def _infer_grid_shape(schema: Mapping[str, Any], dataset_cfg: Mapping[str, Any]) -> tuple[int, int] | None:
    explicit = _tuple2(dataset_cfg.get("grid_shape"))
    if explicit is not None:
        return explicit
    grid_shape = schema.get("grid_shape")
    if grid_shape is not None:
        return _tuple2(grid_shape)
    wedge_length = schema.get("rho_wedge_length")
    if wedge_length is None:
        return None
    try:
        return infer_square_grid_shape_from_wedge_length(int(wedge_length))
    except ValueError:
        return None


@dataclass(frozen=True)
class ForwardRunConfig:
    """Resolved native forward run configuration."""

    source_path: Path
    raw: dict[str, Any]
    experiment_id: str
    dataset_path: Path
    representation: DataRepresentationSpec
    model_name: str
    model_params: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Path | None = None
    runs_dir: Path = Path("runs")
    run_dir: Path | None = None
    epochs: int = 1
    checkpoint_epochs: tuple[int, ...] = field(default_factory=tuple)
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    learning_rates: tuple[float, ...] = field(default_factory=tuple)
    optimizer: str = "adam"
    weight_decay: float = 0.0
    loss_name: str = "mse"
    loss_params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    device: str = "cpu"
    runtime: DeviceConfig = field(default_factory=lambda: DeviceConfig(device="cpu"))
    eval_split: str = "test"
    relative_error_floor: float = 1.0e-8
    training_enabled: bool = True
    execution: str = "train"
    report_path: str | None = None
    seeds: tuple[int, ...] = field(default_factory=lambda: (0,))

    def to_resolved_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable resolved configuration."""

        return {
            "experiment": {"experiment_id": self.experiment_id, "stage": "forward"},
            "dataset": {
                "path": str(self.dataset_path),
                "input_repr": self.representation.input_repr,
                "output_target": self.representation.output_target,
                "grid_shape": list(self.representation.grid_shape)
                if self.representation.grid_shape
                else None,
                "k_points": self.representation.k_points,
                "n_bands": self.representation.n_bands,
            },
            "model": {
                "name": self.model_name,
                "params": self.model_params,
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            },
            "training": {
                "epochs": self.epochs,
                "max_epochs": self.epochs,
                "checkpoint_epochs": list(self.checkpoint_epochs),
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "learning_rates": list(self.learning_rates),
                "optimizer": self.optimizer,
                "weight_decay": self.weight_decay,
                "loss": {"name": self.loss_name, **dict(self.loss_params)},
                "seed": self.seed,
                "seeds": list(self.seeds),
                "device": self.device,
                "enabled": self.training_enabled,
                "execution": self.execution,
            },
            "runtime": self.runtime.to_dict(),
            "evaluation": {"split": self.eval_split},
            "metrics": {"relative_error_floor": self.relative_error_floor},
            "output": {
                "runs_dir": str(self.runs_dir),
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "report": self.report_path,
            },
        }

    def to_resolved_yaml(self) -> str:
        """Serialize the resolved config snapshot."""

        return yaml.safe_dump(self.to_resolved_dict(), sort_keys=False, allow_unicode=True)


def load_forward_run_config(path: str | Path) -> ForwardRunConfig:
    """Load a forward train/eval YAML config and resolve schema-derived shapes."""

    source_path = Path(path)
    raw_payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    raw = _as_mapping(raw_payload, name="config")
    experiment_cfg = _as_mapping(raw.get("experiment"), name="experiment")
    dataset_cfg = _as_mapping(raw.get("dataset"), name="dataset")
    model_cfg = _as_mapping(raw.get("model"), name="model")
    training_cfg = _as_mapping(raw.get("training"), name="training")
    runtime_cfg = _as_mapping(raw.get("runtime"), name="runtime")
    evaluation_cfg = _as_mapping(raw.get("evaluation"), name="evaluation")
    metrics_cfg = _as_mapping(raw.get("metrics"), name="metrics")
    output_cfg = _as_mapping(raw.get("output"), name="output")

    dataset_path = Path(str(dataset_cfg.get("path", "data/tiny.npz")))
    schema = infer_dataset_schema(dataset_path)
    output_target = str(dataset_cfg.get("output_target", "Wh"))
    input_repr = _first_string(
        dataset_cfg.get("input_repr", dataset_cfg.get("input_representations")),
        default="rho_wedge",
    )
    representation = DataRepresentationSpec(
        input_repr=input_repr,  # type: ignore[arg-type]
        output_target=output_target,  # type: ignore[arg-type]
        grid_shape=_infer_grid_shape(schema, dataset_cfg),
        k_points=int(schema["k_points"]) if schema.get("k_points") is not None else None,
        n_bands=int(schema["n_bands"]) if schema.get("n_bands") is not None else None,
    )

    params = {}
    for key in ("params", "overrides"):
        params.update(_as_mapping(model_cfg.get(key), name=f"model.{key}"))

    checkpoint_path = _optional_path(model_cfg.get("checkpoint_path"))
    checkpoint_path = checkpoint_path or _optional_path(evaluation_cfg.get("checkpoint_path"))

    experiment_id = str(
        experiment_cfg.get(
            "experiment_id",
            f"forward_{_first_model_name(model_cfg)}_{input_repr}_{output_target}",
        )
    )
    runs_dir = Path(str(output_cfg.get("runs_dir", "runs")))
    run_dir = _optional_path(output_cfg.get("run_dir"))
    training_enabled = _coerce_bool(training_cfg.get("enabled", True), default=True)
    execution = str(experiment_cfg.get("execution", training_cfg.get("execution", "train")))
    report_path = str(output_cfg.get("report")) if output_cfg.get("report") is not None else None
    loss_cfg = _as_mapping(training_cfg.get("loss"), name="training.loss")
    max_epochs = training_cfg.get("max_epochs", training_cfg.get("epochs", 1))
    raw_checkpoint_epochs = training_cfg.get("checkpoint_epochs", ())
    if raw_checkpoint_epochs is None:
        checkpoint_epochs: tuple[int, ...] = ()
    elif isinstance(raw_checkpoint_epochs, (list, tuple)):
        checkpoint_epochs = tuple(sorted({int(item) for item in raw_checkpoint_epochs if int(item) > 0}))
    else:
        checkpoint_epochs = (int(raw_checkpoint_epochs),)
    raw_learning_rates = training_cfg.get("learning_rates", ())
    if raw_learning_rates is None:
        learning_rates: tuple[float, ...] = ()
    elif isinstance(raw_learning_rates, (list, tuple)):
        learning_rates = tuple(float(item) for item in raw_learning_rates)
    elif raw_learning_rates:
        learning_rates = (float(raw_learning_rates),)
    else:
        learning_rates = ()

    runtime = DeviceConfig.from_mapping(
        runtime_cfg,
        fallback_device=str(training_cfg.get("device", "cpu")),
    )

    return ForwardRunConfig(
        source_path=source_path,
        raw=raw,
        experiment_id=experiment_id,
        dataset_path=dataset_path,
        representation=representation,
        model_name=_first_model_name(model_cfg),
        model_params=params,
        checkpoint_path=checkpoint_path,
        runs_dir=runs_dir,
        run_dir=run_dir,
        epochs=max(1, int(max_epochs)),
        checkpoint_epochs=checkpoint_epochs,
        batch_size=max(1, int(training_cfg.get("batch_size", 64))),
        learning_rate=float(training_cfg.get("learning_rate", training_cfg.get("lr", 1.0e-3))),
        learning_rates=learning_rates,
        optimizer=str(training_cfg.get("optimizer", "adam")).lower(),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        loss_name=str(loss_cfg.get("name", "mse")).lower(),
        loss_params={key: value for key, value in loss_cfg.items() if key != "name"},
        seed=int(training_cfg.get("seed", 0)),
        seeds=tuple(_coerce_seed_list(training_cfg.get("seeds"), default=int(training_cfg.get("seed", 0)))),
        device=str(training_cfg.get("device", "cpu")),
        runtime=runtime,
        eval_split=str(evaluation_cfg.get("split", "test")),
        relative_error_floor=float(metrics_cfg.get("relative_error_floor", 1.0e-8)),
        training_enabled=training_enabled,
        execution=execution,
        report_path=report_path,
    )


__all__ = ["ForwardRunConfig", "load_forward_run_config"]
