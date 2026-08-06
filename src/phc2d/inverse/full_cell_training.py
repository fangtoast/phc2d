"""Full-cell inverse Stage-A data, preflight, and resumable training contracts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.data.full_cell import FULL_CELL_SCHEMA_VERSION, FullCellStructureSpec
from phc2d.inverse.full_cell_codec import (
    FULL_CELL_VARIANTS,
    FullCellAnalyticCodec,
    fit_full_cell_analytic_codecs,
    full_cell_variant_id,
    save_full_cell_analytic_codecs,
)
from phc2d.runs import (
    RUN_LOG_NAME,
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    validate_run_artifacts,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)
from phc2d.training import DeviceConfig, peak_gpu_memory_mb, reset_peak_gpu_memory, resolve_device


FULL_CELL_STAGE_A_METHODS = (
    "direct_inverse_mlp",
    "single_mode_generator",
    "mixture_conditional_generator",
)


def _decode_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def _code_revision() -> str | None:
    deployed = os.environ.get("PHC2D_DEPLOYED_GIT_COMMIT")
    if deployed:
        return deployed.strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in keys:
                keys.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path, *, torch: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


@dataclass(frozen=True)
class FullCellInverseVariantDataset:
    """One variant aligned to global HDF5 rows and frozen route normalization."""

    dataset_path: Path
    variant_id: str
    structures: tuple[FullCellStructureSpec, ...]
    global_indices: np.ndarray
    Wh: np.ndarray
    encoded_parameters: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    condition_mean: np.ndarray
    condition_std: np.ndarray
    codec: FullCellAnalyticCodec

    def normalized_conditions(self, local_indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return flattened route-normalized Wh conditions for local rows."""

        indices = np.asarray(local_indices, dtype=np.int64).reshape(-1)
        values = (self.Wh[indices] - self.condition_mean) / self.condition_std
        return np.asarray(values, dtype=np.float32).reshape(int(indices.size), -1)


def load_full_cell_inverse_variant_datasets(
    path: str | Path,
    *,
    variants: Sequence[str] | None = None,
    require_production_contract: bool = True,
) -> dict[str, FullCellInverseVariantDataset]:
    """Load all requested variants without reading stored raster datasets."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("h5py is required for full-cell inverse training data") from exc

    source = Path(path)
    requested = tuple(str(item) for item in (variants or FULL_CELL_VARIANTS))
    unknown = sorted(set(requested) - set(FULL_CELL_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown full-cell inverse variants: {', '.join(unknown)}")
    with h5py.File(source, "r") as handle:
        if str(handle.attrs.get("schema_version", "")) != FULL_CELL_SCHEMA_VERSION:
            raise ValueError(f"Not a phc2d full-cell v1 HDF5 dataset: {source}")
        required = {"structure_json", "Wh", "idx_train", "idx_val", "idx_test"}
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise ValueError(f"Full-cell HDF5 is missing fields {missing}: {source}")
        structures = tuple(
            FullCellStructureSpec.from_mapping(json.loads(_decode_text(value)))
            for value in handle["structure_json"][:]
        )
        wh = np.asarray(handle["Wh"], dtype=np.float32)
        split_global = {
            "train": np.asarray(handle["idx_train"], dtype=np.int64),
            "val": np.asarray(handle["idx_val"], dtype=np.int64),
            "test": np.asarray(handle["idx_test"], dtype=np.int64),
        }
    if wh.ndim != 3 or wh.shape[0] != len(structures):
        raise ValueError("Full-cell Wh must have shape [n_structures, n_k_points, n_bands]")
    merged = np.concatenate(tuple(split_global.values()))
    if sorted(int(item) for item in merged) != list(range(len(structures))):
        raise ValueError("Full-cell train/val/test indices must partition all structures")

    route_train = wh[split_global["train"]]
    condition_mean = np.mean(route_train, axis=0, dtype=np.float64).astype(np.float32)
    condition_std = np.std(route_train, axis=0, dtype=np.float64).astype(np.float32)
    condition_std = np.maximum(condition_std, np.float32(1.0e-6))
    variant_labels = np.asarray([full_cell_variant_id(item) for item in structures], dtype=object)
    datasets: dict[str, FullCellInverseVariantDataset] = {}
    for variant in requested:
        global_indices = np.flatnonzero(variant_labels == variant).astype(np.int64)
        if global_indices.size == 0:
            continue
        global_to_local = {int(global_index): local for local, global_index in enumerate(global_indices)}
        local_splits = {
            name: np.asarray(
                [global_to_local[int(index)] for index in indices if int(index) in global_to_local],
                dtype=np.int64,
            )
            for name, indices in split_global.items()
        }
        variant_structures = tuple(structures[int(index)] for index in global_indices)
        train_structures = tuple(variant_structures[int(index)] for index in local_splits["train"])
        codec = fit_full_cell_analytic_codecs(
            train_structures,
            require_all_variants=False,
            require_production_contract=require_production_contract,
        )[variant]
        encoded = codec.encode_many(variant_structures).astype(np.float32)
        datasets[variant] = FullCellInverseVariantDataset(
            dataset_path=source,
            variant_id=variant,
            structures=variant_structures,
            global_indices=global_indices,
            Wh=np.asarray(wh[global_indices], dtype=np.float32),
            encoded_parameters=encoded,
            idx_train=local_splits["train"],
            idx_val=local_splits["val"],
            idx_test=local_splits["test"],
            condition_mean=condition_mean,
            condition_std=condition_std,
            codec=codec,
        )
    return datasets


class _AnalyticStageAModel:
    """Torch-lazy direct or mode-conditioned network with an unbounded output head."""

    def __init__(
        self,
        *,
        method: str,
        condition_dim: int,
        output_dim: int,
        hidden_dim: int,
        depth: int,
        latent_dim: int,
        mode_count: int,
        torch: Any,
    ) -> None:
        if method not in FULL_CELL_STAGE_A_METHODS:
            raise ValueError(f"Unknown full-cell Stage-A method: {method}")
        self.method = method
        self.condition_dim = int(condition_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.latent_dim = 0 if method == "direct_inverse_mlp" else int(latent_dim)
        self.mode_count = 1 if method != "mixture_conditional_generator" else int(mode_count)
        nn = torch.nn
        if method == "direct_inverse_mlp":
            layers: list[Any] = []
            for layer_index in range(max(1, self.depth)):
                layers.append(
                    nn.Linear(
                        self.condition_dim if layer_index == 0 else self.hidden_dim,
                        self.hidden_dim,
                    )
                )
                layers.append(nn.GELU())
            layers.append(nn.Linear(self.hidden_dim, self.output_dim))
            self.direct = nn.Sequential(*layers)
            self.condition_encoder = None
            self.mode_embedding = None
            self.decoder = None
        else:
            self.direct = None
            self.condition_encoder = nn.Sequential(
                nn.Linear(self.condition_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
            )
            self.mode_embedding = nn.Embedding(self.mode_count, self.hidden_dim)
            decoder_layers: list[Any] = []
            decoder_input = self.hidden_dim * 2 + self.latent_dim
            for layer_index in range(max(1, self.depth)):
                decoder_layers.append(
                    nn.Linear(
                        decoder_input if layer_index == 0 else self.hidden_dim,
                        self.hidden_dim,
                    )
                )
                decoder_layers.append(nn.GELU())
            decoder_layers.append(nn.Linear(self.hidden_dim, self.output_dim))
            self.decoder = nn.Sequential(*decoder_layers)

    def to(self, device: Any) -> "_AnalyticStageAModel":
        for module in self._modules():
            module.to(device)
        return self

    def train(self) -> None:
        for module in self._modules():
            module.train()

    def eval(self) -> None:
        for module in self._modules():
            module.eval()

    def _modules(self) -> list[Any]:
        return [
            module
            for module in (self.direct, self.condition_encoder, self.mode_embedding, self.decoder)
            if module is not None
        ]

    def parameters(self) -> list[Any]:
        return [parameter for module in self._modules() for parameter in module.parameters()]

    def reconstruct(self, condition: Any, *, torch: Any) -> Any:
        if self.direct is not None:
            return self.direct(condition)
        encoded = self.condition_encoder(condition)
        batch = int(condition.shape[0])
        mode_ids = torch.arange(batch, device=condition.device, dtype=torch.long) % self.mode_count
        mode = self.mode_embedding(mode_ids)
        noise = torch.zeros((batch, self.latent_dim), device=condition.device, dtype=condition.dtype)
        return self.decoder(torch.cat([encoded, mode, noise], dim=1))

    def state_dict(self) -> dict[str, Any]:
        return {name: module.state_dict() for name, module in self._named_modules().items()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name, module in self._named_modules().items():
            module.load_state_dict(dict(state[name]))

    def _named_modules(self) -> dict[str, Any]:
        return {
            name: module
            for name, module in (
                ("direct", self.direct),
                ("condition_encoder", self.condition_encoder),
                ("mode_embedding", self.mode_embedding),
                ("decoder", self.decoder),
            )
            if module is not None
        }


def _unit_circle_loss(prediction: Any, codec: FullCellAnalyticCodec, *, torch: Any) -> Any:
    cursor = 0
    terms: list[Any] = []
    for kind in codec.transform_kinds:
        if kind in {"periodic", "angle"}:
            terms.append(torch.square(torch.square(prediction[:, cursor]) + torch.square(prediction[:, cursor + 1]) - 1.0))
            cursor += 2
        else:
            cursor += 1
    if not terms:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    return torch.mean(torch.stack([torch.mean(term) for term in terms]))


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError("Full-cell inverse Stage-A preflight config must be a mapping")
    return dict(payload)


def _validate_representation(payload: Mapping[str, Any]) -> None:
    representation = dict(payload.get("representation") or {})
    if representation.get("condition_target") != "Wh":
        raise ValueError("representation.condition_target must be Wh")
    if representation.get("output_repr") != "full_cell_analytic_params":
        raise ValueError("representation.output_repr must be full_cell_analytic_params")
    if representation.get("condition_normalization") != "route_train_per_k_band":
        raise ValueError(
            "representation.condition_normalization must be route_train_per_k_band"
        )


def _validated_config(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(config)
    _validate_representation(payload)
    datasets = dict(payload.get("datasets") or {})
    if set(datasets) != {"binary", "graded"}:
        raise ValueError("datasets must define exactly binary and graded routes")
    methods = tuple(str(item) for item in dict(payload.get("models") or {}).get("methods", ()))
    if methods != FULL_CELL_STAGE_A_METHODS:
        raise ValueError(f"models.methods must be exactly {list(FULL_CELL_STAGE_A_METHODS)}")
    training = dict(payload.get("training") or {})
    runtime = DeviceConfig.from_mapping(payload.get("runtime"))
    batch_size = int(training.get("batch_size", 0))
    raster_shape = tuple(int(item) for item in training.get("raster_shape", ()))
    if batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    if len(raster_shape) != 2 or any(item <= 0 for item in raster_shape):
        raise ValueError("training.raster_shape must contain two positive dimensions")
    if runtime.require_accelerator:
        if runtime.device not in {"cuda", "cuda:0"} or runtime.precision != "fp32":
            raise ValueError("Formal N40 preflight requires cuda:0 and fp32")
        if runtime.amp_enabled or runtime.tf32_enabled:
            raise ValueError("Formal N40 preflight forbids AMP and TF32")
        if batch_size != 64 or raster_shape != (64, 64):
            raise ValueError("Formal N40 preflight requires batch_size=64 and raster_shape=[64,64]")
        expected = {
            "direct_inverse_mlp": (512, 2, 0, 1),
            "single_mode_generator": (512, 4, 32, 1),
            "mixture_conditional_generator": (512, 4, 32, 4),
        }
        overrides = dict(dict(payload.get("models") or {}).get("overrides") or {})
        for method, values in expected.items():
            case = dict(overrides.get(method) or {})
            actual = (
                int(case.get("hidden_dim", 0)),
                int(case.get("depth", 0)),
                int(case.get("latent_dim", 0)),
                int(case.get("mode_count", 0)),
            )
            if actual != values:
                raise ValueError(f"Formal N40 network contract mismatch for {method}: {actual} != {values}")
    return payload


def load_full_cell_stage_a_preflight_config(path: str | Path) -> dict[str, Any]:
    """Load and fail-closed validate a CPU or formal N40 preflight config."""

    return _validated_config(_load_config(Path(path)))


def _validated_train_config(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = _validated_config(config)
    experiment = dict(payload.get("experiment") or {})
    if experiment.get("stage") != "inverse_full_cell_stage_a_train":
        raise ValueError("experiment.stage must be inverse_full_cell_stage_a_train")
    training = dict(payload.get("training") or {})
    epochs = int(training.get("epochs", 0))
    val_batch_size = int(training.get("val_batch_size", 0))
    patience = int(training.get("early_stopping_patience", 0))
    checkpoint_epochs = tuple(int(item) for item in training.get("checkpoint_epochs", ()))
    if epochs <= 0 or val_batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, val_batch_size, and early_stopping_patience must be positive")
    if any(item <= 0 or item > epochs for item in checkpoint_epochs):
        raise ValueError("checkpoint_epochs must be positive and no larger than epochs")
    learning_rates = dict(training.get("learning_rates") or {})
    if set(learning_rates) != set(FULL_CELL_STAGE_A_METHODS):
        raise ValueError("training.learning_rates must cover all three Stage-A methods")
    if any(float(value) <= 0.0 for value in learning_rates.values()):
        raise ValueError("training learning rates must be positive")
    forward_surrogate_status = str(
        dict(payload.get("gates") or {}).get("forward_surrogate_status", "")
    )
    if forward_surrogate_status not in {"not_trained", "accepted"}:
        raise ValueError(
            "gates.forward_surrogate_status must be explicitly not_trained or accepted"
        )
    runtime = DeviceConfig.from_mapping(payload.get("runtime"))
    if runtime.require_accelerator:
        if epochs != 200 or checkpoint_epochs != (50, 100, 150, 200) or patience != 20:
            raise ValueError(
                "Formal N40 Stage-A training requires epochs=200, checkpoints=50/100/150/200, patience=20"
            )
        expected_rates = {
            "direct_inverse_mlp": 3.0e-4,
            "single_mode_generator": 1.0e-4,
            "mixture_conditional_generator": 1.0e-4,
        }
        if any(
            not np.isclose(float(learning_rates[key]), value, rtol=0.0, atol=1.0e-12)
            for key, value in expected_rates.items()
        ):
            raise ValueError("Formal N40 Stage-A learning-rate contract mismatch")
        datasets = dict(payload["datasets"])
        for route in ("binary", "graded"):
            expected_hash = str(dict(datasets[route]).get("expected_dataset_sha256", ""))
            if len(expected_hash) != 64:
                raise ValueError(
                    f"Formal N40 Stage-A training requires datasets.{route}.expected_dataset_sha256"
                )
    return payload


def load_full_cell_stage_a_train_config(path: str | Path) -> dict[str, Any]:
    """Load and fail-closed validate a Stage-A epoch training config."""

    return _validated_train_config(_load_config(Path(path)))


def run_full_cell_stage_a_preflight(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
) -> Path:
    """Run all 15 single-batch Stage-A cases and write auditable artifacts."""

    config_file = Path(config_path)
    config = load_full_cell_stage_a_preflight_config(config_file)
    experiment = dict(config.get("experiment") or {})
    dataset_cfg = dict(config["datasets"])
    models_cfg = dict(config.get("models") or {})
    training_cfg = dict(config.get("training") or {})
    output_cfg = dict(config.get("output") or {})
    runtime = DeviceConfig.from_mapping(config.get("runtime"))
    task = str(experiment.get("experiment_id", "full_cell_inverse_stage_a_preflight"))
    out = make_run_dir(task, runs_dir=output_cfg.get("runs_dir", "runs"), run_dir=run_dir)
    logger = setup_logging(
        log_file=out / RUN_LOG_NAME,
        run_meta={
            "script": "phc2d inverse full-cell-preflight",
            "config_path": str(config_file),
            "validation_boundary": "full_cell_inverse_stage_a_single_batch_no_formal_training_no_mpb",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
        },
    )
    write_resolved_config(out, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    ensure_figures_dir(out)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("torch is required for full-cell Stage-A preflight") from exc
    if int(dict(config.get("runtime") or {}).get("cpu_threads", 0)) > 0:
        torch.set_num_threads(int(dict(config.get("runtime") or {})["cpu_threads"]))
    device_info = resolve_device(runtime, torch_module=torch)
    device = device_info.torch_device
    batch_size = int(training_cfg["batch_size"])
    raster_shape = tuple(int(item) for item in training_cfg["raster_shape"])
    binary_temperature = float(training_cfg.get("binary_temperature", 0.01))
    raster_weight = float(training_cfg.get("raster_weight", 0.1))
    unit_circle_weight = float(training_cfg.get("unit_circle_weight", 0.05))
    learning_rate = float(training_cfg.get("learning_rate", 1.0e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1.0e-5))
    seed = int(training_cfg.get("seed", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    wall_start = time.perf_counter()
    route_variants = {
        "binary": FULL_CELL_VARIANTS[:3],
        "graded": FULL_CELL_VARIANTS[3:],
    }
    route_data: dict[str, dict[str, FullCellInverseVariantDataset]] = {}
    dataset_contracts: dict[str, Any] = {}
    all_codecs: dict[str, FullCellAnalyticCodec] = {}
    for route, variants in route_variants.items():
        path = Path(str(dict(dataset_cfg[route])["path"]))
        loaded = load_full_cell_inverse_variant_datasets(path, variants=variants)
        if tuple(loaded) != tuple(variants):
            raise ValueError(f"Dataset route {route} does not contain its frozen variants")
        route_data[route] = loaded
        all_codecs.update({key: value.codec for key, value in loaded.items()})
        representative = next(iter(loaded.values()))
        dataset_contracts[route] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "Wh_shape": list(representative.condition_mean.shape),
            "condition_mean_sha256": _sha256_array(representative.condition_mean),
            "condition_std_sha256": _sha256_array(representative.condition_std),
            "variants": {
                key: {
                    "samples": len(value.structures),
                    "train": int(value.idx_train.size),
                    "val": int(value.idx_val.size),
                    "test": int(value.idx_test.size),
                    "physical_width": value.codec.physical_width,
                    "encoded_width": value.codec.encoded_width,
                }
                for key, value in loaded.items()
            },
        }
        np.savez(
            out / f"condition_normalization_{route}.npz",
            mean=representative.condition_mean,
            std=representative.condition_std,
        )
    save_full_cell_analytic_codecs(all_codecs, out / "codec.resolved.json")

    overrides = dict(models_cfg.get("overrides") or {})
    methods = tuple(str(item) for item in models_cfg["methods"])
    history: list[dict[str, Any]] = []
    checkpoint_paths: list[str] = []
    for route, datasets in route_data.items():
        for variant, dataset in datasets.items():
            local_indices = dataset.idx_train[:batch_size]
            if int(local_indices.size) != batch_size:
                raise ValueError(f"Variant {variant} has fewer than batch_size={batch_size} train rows")
            condition_np = dataset.normalized_conditions(local_indices)
            target_np = dataset.encoded_parameters[local_indices]
            condition = torch.as_tensor(condition_np, dtype=torch.float32, device=device)
            target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
            reference_raster = dataset.codec.decode_soft_raster_tensor(
                target,
                grid_shape=raster_shape,
                binary_temperature=binary_temperature,
                torch=torch,
            ).detach()
            for method in methods:
                case_cfg = dict(overrides.get(method) or {})
                model_args = {
                    "method": method,
                    "condition_dim": int(condition.shape[1]),
                    "output_dim": dataset.codec.encoded_width,
                    "hidden_dim": int(case_cfg.get("hidden_dim", 512)),
                    "depth": int(case_cfg.get("depth", 2 if method == "direct_inverse_mlp" else 4)),
                    "latent_dim": int(case_cfg.get("latent_dim", 0 if method == "direct_inverse_mlp" else 32)),
                    "mode_count": int(case_cfg.get("mode_count", 4 if method == "mixture_conditional_generator" else 1)),
                }
                torch.manual_seed(seed)
                model = _AnalyticStageAModel(**model_args, torch=torch).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
                reset_peak_gpu_memory(device, torch_module=torch)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                case_start = time.perf_counter()
                prediction = model.reconstruct(condition, torch=torch)
                predicted_raster = dataset.codec.decode_soft_raster_tensor(
                    prediction,
                    grid_shape=raster_shape,
                    binary_temperature=binary_temperature,
                    torch=torch,
                )
                encoded_loss = torch.mean(torch.square(prediction - target))
                raster_loss = torch.mean(torch.square(predicted_raster - reference_raster))
                circle_loss = _unit_circle_loss(prediction, dataset.codec, torch=torch)
                total_loss = encoded_loss + raster_weight * raster_loss + unit_circle_weight * circle_loss
                total_loss.backward()
                gradients = [parameter.grad.reshape(-1) for parameter in model.parameters() if parameter.grad is not None]
                if not gradients:
                    raise RuntimeError(f"No gradients produced for {variant}/{method}")
                gradient_vector = torch.cat(gradients)
                gradient_finite = bool(torch.all(torch.isfinite(gradient_vector)).item())
                gradient_norm = float(torch.linalg.vector_norm(gradient_vector).detach().cpu())
                if not gradient_finite or gradient_norm <= 0.0 or not bool(torch.isfinite(total_loss).item()):
                    raise RuntimeError(f"Invalid Stage-A backward result for {variant}/{method}")
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    saved_prediction = model.reconstruct(condition, torch=torch).detach().cpu()
                case_dir = out / "cases" / variant / method
                case_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = case_dir / "preflight.pt"
                _atomic_torch_save(
                    {
                        "stage": "full_cell_inverse_stage_a_preflight",
                        "variant_id": variant,
                        "method": method,
                        "model_args": model_args,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "codec": dataset.codec.to_mapping(),
                        "condition_mean": dataset.condition_mean,
                        "condition_std": dataset.condition_std,
                        "dataset_path": str(dataset.dataset_path),
                        "dataset_sha256": dataset_contracts[route]["sha256"],
                        "global_indices": dataset.global_indices[local_indices],
                        "seed": seed,
                    },
                    checkpoint,
                    torch=torch,
                )
                try:
                    restored_payload = torch.load(checkpoint, map_location=device, weights_only=False)
                except TypeError:  # pragma: no cover - older torch
                    restored_payload = torch.load(checkpoint, map_location=device)
                restored = _AnalyticStageAModel(**dict(restored_payload["model_args"]), torch=torch).to(device)
                restored.load_state_dict(restored_payload["state_dict"])
                restored.eval()
                with torch.no_grad():
                    restored_prediction = restored.reconstruct(condition, torch=torch).detach().cpu()
                checkpoint_match = bool(torch.equal(saved_prediction, restored_prediction))
                if not checkpoint_match:
                    raise RuntimeError(f"Checkpoint reload mismatch for {variant}/{method}")
                elapsed = float(time.perf_counter() - case_start)
                row = {
                    "route": route,
                    "variant_id": variant,
                    "method": method,
                    "batch_size": batch_size,
                    "condition_dim": int(condition.shape[1]),
                    "encoded_width": dataset.codec.encoded_width,
                    "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                    "encoded_loss": float(encoded_loss.detach().cpu()),
                    "raster_loss": float(raster_loss.detach().cpu()),
                    "unit_circle_loss": float(circle_loss.detach().cpu()),
                    "total_loss": float(total_loss.detach().cpu()),
                    "gradient_norm": gradient_norm,
                    "gradient_finite": gradient_finite,
                    "checkpoint_reload_exact_match": checkpoint_match,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": _sha256_file(checkpoint),
                    "peak_gpu_memory_mb": peak_gpu_memory_mb(device, torch_module=torch) or 0.0,
                    "wall_time_s": elapsed,
                }
                history.append(row)
                checkpoint_paths.append(str(checkpoint))
                logger.info(
                    "Stage-A preflight pass %s/%s loss=%.6g grad=%.6g",
                    variant,
                    method,
                    row["total_loss"],
                    gradient_norm,
                )

    expected_cases = len(FULL_CELL_VARIANTS) * len(FULL_CELL_STAGE_A_METHODS)
    if len(history) != expected_cases:
        raise RuntimeError(f"Expected {expected_cases} Stage-A cases, got {len(history)}")
    wall_time = float(time.perf_counter() - wall_start)
    status = "n40_single_card_preflight_pass" if device_info.is_cuda else "cpu_contract_preflight_pass"
    metrics = {
        "status": status,
        "validation_boundary": "full_cell_inverse_stage_a_single_batch_no_formal_training_no_mpb",
        "case_count": len(history),
        "batch_size": batch_size,
        "raster_shape": list(raster_shape),
        "all_gradients_finite_nonzero": all(
            bool(row["gradient_finite"]) and float(row["gradient_norm"]) > 0.0 for row in history
        ),
        "all_checkpoint_reloads_exact": all(bool(row["checkpoint_reload_exact_match"]) for row in history),
        "wall_time_total_s": wall_time,
        **device_info.metadata(peak_gpu_memory_mb=peak_gpu_memory_mb(device, torch_module=torch)),
    }
    _write_history(out / "history.csv", history)
    write_metrics(out, metrics)
    write_results(
        out,
        {
            "status": status,
            "formal_training_started": False,
            "mpb_meep_used": False,
            "dataset_contracts": dataset_contracts,
            "case_results": history,
            "checkpoint_paths": checkpoint_paths,
            "next_gate": "n40_single_card_preflight_authorization"
            if not device_info.is_cuda
            else "n40_18_job_concurrency_admission",
        },
    )
    write_run_meta(
        out,
        {
            "script": "phc2d inverse full-cell-preflight",
            "config_path": str(config_file),
            "validation_boundary": metrics["validation_boundary"],
            "status": status,
            "wall_time_total_s": wall_time,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
        },
    )
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"Full-cell Stage-A preflight artifacts are incomplete: {missing}")
    logger.info("Full-cell Stage-A preflight passed: %s", out)
    return out


def _stage_a_batch_losses(
    model: _AnalyticStageAModel,
    condition: Any,
    target: Any,
    *,
    codec: FullCellAnalyticCodec,
    raster_shape: tuple[int, int],
    binary_temperature: float,
    raster_weight: float,
    unit_circle_weight: float,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    prediction = model.reconstruct(condition, torch=torch)
    predicted_raster = codec.decode_soft_raster_tensor(
        prediction,
        grid_shape=raster_shape,
        binary_temperature=binary_temperature,
        torch=torch,
    )
    reference_raster = codec.decode_soft_raster_tensor(
        target,
        grid_shape=raster_shape,
        binary_temperature=binary_temperature,
        torch=torch,
    ).detach()
    encoded_loss = torch.mean(torch.square(prediction - target))
    raster_loss = torch.mean(torch.square(predicted_raster - reference_raster))
    circle_loss = _unit_circle_loss(prediction, codec, torch=torch)
    total_loss = encoded_loss + raster_weight * raster_loss + unit_circle_weight * circle_loss
    return total_loss, {
        "encoded_loss": encoded_loss,
        "raster_loss": raster_loss,
        "unit_circle_loss": circle_loss,
        "total_loss": total_loss,
    }


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _plot_stage_a_history(run_dir: Path, history: Sequence[Mapping[str, Any]]) -> None:
    if not history:
        return
    import matplotlib.pyplot as plt

    epochs = np.asarray([int(row["epoch"]) for row in history], dtype=np.int64)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    for split, style in (("train", "-"), ("val", "--")):
        axes[0].plot(
            epochs,
            [float(row[f"{split}_total_loss"]) for row in history],
            style,
            label=f"{split} total",
        )
        axes[1].plot(
            epochs,
            [float(row[f"{split}_raster_loss"]) for row in history],
            style,
            label=f"{split} raster",
        )
    best_rows = [row for row in history if str(row.get("is_best", "")).lower() == "true"]
    if best_rows:
        best = best_rows[-1]
        axes[0].scatter(
            [int(best["epoch"])],
            [float(best["val_total_loss"])],
            marker="*",
            s=80,
            label="best",
        )
    axes[0].set_title("Stage-A composite loss")
    axes[1].set_title("Decoded-raster loss")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend()
    for suffix in ("png", "pdf"):
        figure.savefig(run_dir / "figures" / f"training_curve.{suffix}", dpi=180)
    plt.close(figure)


def _mean_loss_rows(rows: Sequence[Mapping[str, float]], prefix: str) -> dict[str, float]:
    if not rows:
        raise RuntimeError(f"No {prefix} batches were evaluated")
    keys = ("encoded_loss", "raster_loss", "unit_circle_loss", "total_loss")
    total_weight = float(sum(float(row["batch_size"]) for row in rows))
    return {
        f"{prefix}_{key}": float(
            sum(float(row[key]) * float(row["batch_size"]) for row in rows) / total_weight
        )
        for key in keys
    }


def _codec_sha256(codec: FullCellAnalyticCodec) -> str:
    encoded = json.dumps(
        codec.to_mapping(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_full_cell_stage_a_train(
    config_path: str | Path,
    *,
    variant: str,
    method: str,
    seed: int,
    run_dir: str | Path | None = None,
    stop_after_epoch: int | None = None,
) -> Path:
    """Train one full-cell Stage-A variant/method/seed with validation and resume."""

    config_file = Path(config_path)
    config = load_full_cell_stage_a_train_config(config_file)
    if variant not in FULL_CELL_VARIANTS:
        raise ValueError(f"Unknown full-cell inverse variant: {variant}")
    if method not in FULL_CELL_STAGE_A_METHODS:
        raise ValueError(f"Unknown full-cell Stage-A method: {method}")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    experiment = dict(config.get("experiment") or {})
    datasets_cfg = dict(config["datasets"])
    models_cfg = dict(config["models"])
    training = dict(config["training"])
    runtime = DeviceConfig.from_mapping(config.get("runtime"))
    output = dict(config.get("output") or {})
    route = "binary" if variant.startswith("binary_") else "graded"
    dataset_path = Path(str(dict(datasets_cfg[route])["path"]))
    task = (
        f"{str(experiment.get('experiment_id', 'full_cell_inverse_stage_a'))}_"
        f"{variant}_{method}_seed{int(seed)}"
    )
    out = make_run_dir(task, runs_dir=output.get("runs_dir", "runs"), run_dir=run_dir)
    validation_boundary = "full_cell_inverse_stage_a_train_no_stage_b_no_mpb"
    logger = setup_logging(
        log_file=out / RUN_LOG_NAME,
        run_meta={
            "script": "phc2d inverse full-cell-train",
            "config_path": str(config_file),
            "dataset_path": str(dataset_path),
            "variant_id": variant,
            "method": method,
            "seed": int(seed),
            "validation_boundary": validation_boundary,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
        },
    )
    write_resolved_config(out, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    ensure_figures_dir(out)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("torch is required for full-cell Stage-A training") from exc
    runtime_raw = dict(config.get("runtime") or {})
    if int(runtime_raw.get("cpu_threads", 0)) > 0:
        torch.set_num_threads(int(runtime_raw["cpu_threads"]))
    device_info = resolve_device(runtime, torch_module=torch)
    device = device_info.torch_device
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    data = load_full_cell_inverse_variant_datasets(
        dataset_path,
        variants=(variant,),
    )[variant]
    config_sha256 = _sha256_file(config_file)
    dataset_sha256 = _sha256_file(dataset_path)
    expected_dataset_sha256 = str(
        dict(datasets_cfg[route]).get("expected_dataset_sha256", "")
    ).lower()
    if expected_dataset_sha256 and dataset_sha256.lower() != expected_dataset_sha256:
        raise ValueError(
            f"Dataset SHA-256 mismatch for {route}: {dataset_sha256} != {expected_dataset_sha256}"
        )
    codec_sha256 = _codec_sha256(data.codec)
    revision = _code_revision()
    identity = {
        "variant_id": variant,
        "method": method,
        "seed": int(seed),
        "config_sha256": config_sha256,
        "dataset_sha256": dataset_sha256,
        "codec_sha256": codec_sha256,
        "code_revision": revision,
    }
    model_cfg = dict(dict(models_cfg.get("overrides") or {}).get(method) or {})
    condition_dim = int(np.prod(data.Wh.shape[1:]))
    model_args = {
        "method": method,
        "condition_dim": condition_dim,
        "output_dim": data.codec.encoded_width,
        "hidden_dim": int(model_cfg.get("hidden_dim", 512)),
        "depth": int(model_cfg.get("depth", 2 if method == "direct_inverse_mlp" else 4)),
        "latent_dim": int(model_cfg.get("latent_dim", 0 if method == "direct_inverse_mlp" else 32)),
        "mode_count": int(model_cfg.get("mode_count", 4 if method == "mixture_conditional_generator" else 1)),
    }
    model = _AnalyticStageAModel(**model_args, torch=torch).to(device)
    learning_rate = float(dict(training["learning_rates"])[method])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training.get("weight_decay", 1.0e-5)),
    )
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    val_batch_size = int(training["val_batch_size"])
    patience = int(training["early_stopping_patience"])
    checkpoint_epochs = {int(item) for item in training.get("checkpoint_epochs", ())}
    raster_shape = tuple(int(item) for item in training["raster_shape"])
    binary_temperature = float(training.get("binary_temperature", 0.01))
    raster_weight = float(training.get("raster_weight", 0.1))
    unit_circle_weight = float(training.get("unit_circle_weight", 0.05))
    forward_surrogate_status = str(
        dict(config.get("gates") or {}).get("forward_surrogate_status", "not_trained")
    )
    history_path = out / "history.csv"
    history = _read_history(history_path)
    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    resumed = False
    last_checkpoint = out / "last.pt"
    if last_checkpoint.is_file():
        try:
            payload = torch.load(last_checkpoint, map_location=device, weights_only=False)
        except TypeError:  # pragma: no cover - older torch
            payload = torch.load(last_checkpoint, map_location=device)
        for key, expected in identity.items():
            if payload.get(key) != expected:
                raise ValueError(
                    f"Resume identity mismatch for {key}: {payload.get(key)!r} != {expected!r}"
                )
        if dict(payload.get("model_args") or {}) != model_args:
            raise ValueError("Resume model_args do not match the frozen Stage-A model")
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_val = float(payload["best_val_total_loss"])
        best_epoch = int(payload["best_epoch"])
        stale_epochs = int(payload["stale_epochs"])
        history = [row for row in history if int(row["epoch"]) < start_epoch]
        resumed = True
        logger.info("Resuming Stage-A run from epoch %d", start_epoch - 1)

    train_conditions = torch.as_tensor(
        data.normalized_conditions(data.idx_train),
        dtype=torch.float32,
    )
    train_targets = torch.as_tensor(data.encoded_parameters[data.idx_train], dtype=torch.float32)
    val_conditions = torch.as_tensor(
        data.normalized_conditions(data.idx_val),
        dtype=torch.float32,
    )
    val_targets = torch.as_tensor(data.encoded_parameters[data.idx_val], dtype=torch.float32)
    if int(train_conditions.shape[0]) < batch_size or int(val_conditions.shape[0]) == 0:
        raise ValueError("Stage-A train/validation split is too small for the configured batches")
    reset_peak_gpu_memory(device, torch_module=torch)
    wall_start = time.perf_counter()
    completed = False
    stop_reason = "max_epochs"
    epoch_limit = epochs if stop_after_epoch is None else min(epochs, int(stop_after_epoch))
    for epoch in range(start_epoch, epoch_limit + 1):
        epoch_start = time.perf_counter()
        order_generator = torch.Generator(device="cpu")
        order_generator.manual_seed(int(seed) * 1_000_003 + epoch)
        order = torch.randperm(int(train_conditions.shape[0]), generator=order_generator)
        train_rows: list[dict[str, float]] = []
        model.train()
        train_start = time.perf_counter()
        for offset in range(0, int(order.numel()), batch_size):
            indices = order[offset : offset + batch_size]
            condition = train_conditions.index_select(0, indices).to(
                device,
                non_blocking=device_info.non_blocking_transfer,
            )
            target = train_targets.index_select(0, indices).to(
                device,
                non_blocking=device_info.non_blocking_transfer,
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss, components = _stage_a_batch_losses(
                model,
                condition,
                target,
                codec=data.codec,
                raster_shape=raster_shape,
                binary_temperature=binary_temperature,
                raster_weight=raster_weight,
                unit_circle_weight=unit_circle_weight,
                torch=torch,
            )
            total_loss.backward()
            gradients = [
                parameter.grad.reshape(-1)
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not gradients:
                raise RuntimeError("Stage-A training produced no gradients")
            gradient_vector = torch.cat(gradients)
            if not bool(torch.all(torch.isfinite(gradient_vector)).item()):
                raise RuntimeError(f"Non-finite Stage-A gradient at epoch {epoch}")
            optimizer.step()
            train_rows.append(
                {
                    **{key: float(value.detach().cpu()) for key, value in components.items()},
                    "batch_size": float(condition.shape[0]),
                }
            )
        epoch_train_time = float(time.perf_counter() - train_start)

        val_rows: list[dict[str, float]] = []
        model.eval()
        eval_start = time.perf_counter()
        with torch.no_grad():
            for offset in range(0, int(val_conditions.shape[0]), val_batch_size):
                condition = val_conditions[offset : offset + val_batch_size].to(
                    device,
                    non_blocking=device_info.non_blocking_transfer,
                )
                target = val_targets[offset : offset + val_batch_size].to(
                    device,
                    non_blocking=device_info.non_blocking_transfer,
                )
                _, components = _stage_a_batch_losses(
                    model,
                    condition,
                    target,
                    codec=data.codec,
                    raster_shape=raster_shape,
                    binary_temperature=binary_temperature,
                    raster_weight=raster_weight,
                    unit_circle_weight=unit_circle_weight,
                    torch=torch,
                )
                val_rows.append(
                    {
                        **{key: float(value.detach().cpu()) for key, value in components.items()},
                        "batch_size": float(condition.shape[0]),
                    }
                )
        epoch_eval_time = float(time.perf_counter() - eval_start)
        epoch_metrics = {
            **_mean_loss_rows(train_rows, "train"),
            **_mean_loss_rows(val_rows, "val"),
        }
        current_val = float(epoch_metrics["val_total_loss"])
        is_best = current_val < best_val
        if is_best:
            best_val = current_val
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "stage": "stage_a",
            "variant_id": variant,
            "method": method,
            "seed": int(seed),
            "epoch": epoch,
            "learning_rate": learning_rate,
            **epoch_metrics,
            "is_best": is_best,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "train_time_s": epoch_train_time,
            "eval_time_s": epoch_eval_time,
            "epoch_time_s": float(time.perf_counter() - epoch_start),
        }
        history.append(row)
        _write_history(history_path, history)
        checkpoint_payload = {
            **identity,
            "stage": "full_cell_inverse_stage_a",
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_total_loss": best_val,
            "stale_epochs": stale_epochs,
            "model_args": model_args,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "learning_rate": learning_rate,
            "condition_mean": data.condition_mean,
            "condition_std": data.condition_std,
            "codec": data.codec.to_mapping(),
            "train_global_indices_sha256": _sha256_array(data.global_indices[data.idx_train]),
            "val_global_indices_sha256": _sha256_array(data.global_indices[data.idx_val]),
            "test_split_touched": False,
        }
        _atomic_torch_save(checkpoint_payload, last_checkpoint, torch=torch)
        if epoch in checkpoint_epochs:
            _atomic_torch_save(
                checkpoint_payload,
                out / f"checkpoint_epoch_{epoch}.pt",
                torch=torch,
            )
        if is_best:
            _atomic_torch_save(checkpoint_payload, out / "best.pt", torch=torch)
        logger.info(
            "Stage-A epoch %d/%d train=%.6g val=%.6g best=%d stale=%d",
            epoch,
            epochs,
            float(epoch_metrics["train_total_loss"]),
            current_val,
            best_epoch,
            stale_epochs,
        )
        if stale_epochs >= patience:
            completed = True
            stop_reason = "early_stopping"
            break
    if stop_after_epoch is None and not completed:
        completed = bool(history and int(history[-1]["epoch"]) >= epochs)
    if stop_after_epoch is not None and epoch_limit < epochs:
        stop_reason = "requested_partial_stop"

    checkpoint_reload_exact = False
    model_path = out / "model.pt"
    if completed:
        best_path = out / "best.pt"
        if not best_path.is_file():
            raise RuntimeError("Stage-A best checkpoint was not written")
        try:
            best_payload = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:  # pragma: no cover - older torch
            best_payload = torch.load(best_path, map_location=device)
        _atomic_torch_save(best_payload, model_path, torch=torch)
        restored = _AnalyticStageAModel(**dict(best_payload["model_args"]), torch=torch).to(device)
        restored.load_state_dict(best_payload["state_dict"])
        restored.eval()
        model.load_state_dict(best_payload["state_dict"])
        model.eval()
        probe = val_conditions[: min(val_batch_size, int(val_conditions.shape[0]))].to(device)
        with torch.no_grad():
            original_prediction = model.reconstruct(probe, torch=torch).detach().cpu()
            restored_prediction = restored.reconstruct(probe, torch=torch).detach().cpu()
        checkpoint_reload_exact = bool(torch.equal(original_prediction, restored_prediction))
        if not checkpoint_reload_exact:
            raise RuntimeError("Final Stage-A checkpoint reload is not exact")

    _plot_stage_a_history(out, history)
    wall_time = float(time.perf_counter() - wall_start)
    train_time = float(sum(float(row["train_time_s"]) for row in history))
    eval_time = float(sum(float(row["eval_time_s"]) for row in history))
    status = "stage_a_train_complete" if completed else "stage_a_train_partial"
    checkpoint_files = sorted(
        path for path in out.glob("*.pt") if path.name != "last.pt" or not completed
    )
    metrics = {
        "status": status,
        "validation_boundary": validation_boundary,
        "variant_id": variant,
        "method": method,
        "seed": int(seed),
        "epochs_completed": int(history[-1]["epoch"]) if history else 0,
        "best_epoch": best_epoch,
        "best_val_total_loss": best_val,
        "stop_reason": stop_reason,
        "resumed_from_checkpoint": resumed,
        "checkpoint_reload_exact_match": checkpoint_reload_exact if completed else None,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
        "wall_time_total_s": wall_time,
        "train_samples_per_s": (
            float(data.idx_train.size * len(history) / train_time) if train_time > 0.0 else 0.0
        ),
        **device_info.metadata(peak_gpu_memory_mb=peak_gpu_memory_mb(device, torch_module=torch)),
    }
    write_metrics(out, metrics)
    write_results(
        out,
        {
            "status": status,
            "formal_training_started": bool(runtime.require_accelerator),
            "mpb_meep_used": False,
            "variant_id": variant,
            "method": method,
            "seed": int(seed),
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_sha256,
            "config_sha256": config_sha256,
            "codec_sha256": codec_sha256,
            "condition_normalization": "route_train_per_k_band",
            "train_count": int(data.idx_train.size),
            "val_count": int(data.idx_val.size),
            "test_count": int(data.idx_test.size),
            "test_split_touched": False,
            "forward_surrogate_status": forward_surrogate_status,
            "stage_b_admitted": False,
            "candidate_ranking_admitted": False,
            "inverse_mpb_replay_admitted": False,
            "best_checkpoint": str(model_path) if completed else None,
            "best_checkpoint_sha256": _sha256_file(model_path) if completed else None,
            "checkpoint_files": [str(path) for path in checkpoint_files],
            "next_gate": (
                "forward_surrogate_acceptance_then_stage_b_authorization"
                if completed
                else "resume_same_run_dir"
            ),
        },
    )
    write_run_meta(
        out,
        {
            "script": "phc2d inverse full-cell-train",
            "config_path": str(config_file),
            "dataset_path": str(dataset_path),
            "variant_id": variant,
            "method": method,
            "seed": int(seed),
            "status": status,
            "validation_boundary": validation_boundary,
            "wall_time_total_s": wall_time,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
        },
    )
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"Full-cell Stage-A run artifacts are incomplete: {missing}")
    logger.info("Full-cell Stage-A run %s: %s", status, out)
    return out


__all__ = [
    "FULL_CELL_STAGE_A_METHODS",
    "FullCellInverseVariantDataset",
    "load_full_cell_stage_a_preflight_config",
    "load_full_cell_stage_a_train_config",
    "load_full_cell_inverse_variant_datasets",
    "run_full_cell_stage_a_preflight",
    "run_full_cell_stage_a_train",
]
