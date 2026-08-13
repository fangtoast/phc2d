"""Frozen and mock surrogate adapters for inverse-design ranking.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from phc2d.data import DataRepresentationSpec, load_forward_dataset, make_input, recover_k_path
from phc2d.data.lift_torch import epsilon_from_rho_torch, lift_rho_to_grid_torch
from phc2d.inverse.config import GenerativeInverseConfig
from phc2d.inverse.runtime_support import _as_mapping, _read_csv_rows, _read_yaml
from phc2d.models import build_model, get_model_spec


class _MockSurrogate:
    """Dependency-light forward surrogate for smoke ranking and replay."""

    def __init__(
        self,
        input_dim: int,
        target_shape: tuple[int, ...],
        seed: int,
        *,
        torch: Any | None = None,
        device: Any | None = None,
    ) -> None:
        target_size = int(np.prod(target_shape))
        rng = np.random.default_rng(seed)
        self._weights = rng.normal(0.0, 0.5, size=(int(input_dim), target_size)).astype(np.float32)
        self._bias = rng.normal(0.0, 0.05, size=target_size).astype(np.float32)
        self._target_shape = tuple(int(v) for v in target_shape)
        self.torch = torch
        self.device = device
        if torch is not None and device is not None:
            self._weights_tensor = torch.as_tensor(self._weights, dtype=torch.float32, device=device)
            self._bias_tensor = torch.as_tensor(self._bias, dtype=torch.float32, device=device)
        else:
            self._weights_tensor = None
            self._bias_tensor = None

    def predict(self, candidates: np.ndarray) -> np.ndarray:
        flat = np.asarray(candidates, dtype=np.float32).reshape(int(candidates.shape[0]), -1)
        if flat.shape[1] < self._weights.shape[0]:
            pad = np.zeros((flat.shape[0], self._weights.shape[0] - flat.shape[1]), dtype=np.float32)
            flat = np.concatenate([flat, pad], axis=1)
        elif flat.shape[1] > self._weights.shape[0]:
            flat = flat[:, : self._weights.shape[0]]
        pred = flat @ self._weights + self._bias
        return pred.reshape(int(pred.shape[0]), *self._target_shape)

    def predict_tensor(self, candidates: Any) -> Any:
        if self.torch is None or self.device is None or self._weights_tensor is None or self._bias_tensor is None:
            raise RuntimeError("predict_tensor requires torch and device")
        flat = candidates.reshape(int(candidates.shape[0]), -1).to(dtype=self.torch.float32, device=self.device)
        width = int(self._weights_tensor.shape[0])
        if int(flat.shape[1]) < width:
            pad = self.torch.zeros((int(flat.shape[0]), width - int(flat.shape[1])), dtype=self.torch.float32, device=self.device)
            flat = self.torch.cat([flat, pad], dim=1)
        elif int(flat.shape[1]) > width:
            flat = flat[:, :width]
        pred = flat @ self._weights_tensor + self._bias_tensor
        return pred.reshape(int(pred.shape[0]), *self._target_shape)


class _QueryFrozenSurrogate:
    """Frozen query surrogate that predicts full ``Wh`` by sweeping physical k points."""

    def __init__(self, cfg: GenerativeInverseConfig, *, torch: Any, device: Any) -> None:
        surrogate_cfg = _as_mapping(cfg.surrogate)
        checkpoint_path = _resolve_query_surrogate_checkpoint(cfg)
        if checkpoint_path is None:
            raise ValueError("Formal inverse ranking requires surrogate.checkpoint_path or a completed N=1000 query run")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        resolved = _as_mapping(payload.get("resolved_config"))
        configured_path = surrogate_cfg.get("config_path")
        raw = _read_yaml(Path(str(configured_path))) if configured_path else resolved
        dataset_cfg = _as_mapping(raw.get("dataset"))
        model_cfg = _as_mapping(raw.get("model"))
        params = {}
        params.update(_as_mapping(model_cfg.get("params")))
        params.update(_as_mapping(model_cfg.get("overrides")))
        params_by_model = _as_mapping(model_cfg.get("params_by_model"))
        checkpoint_model_name = str(
            surrogate_cfg.get("model_name") or payload.get("model_name") or "fno2d_query"
        )
        params.update(_as_mapping(params_by_model.get(checkpoint_model_name)))

        self.cfg = cfg
        self.torch = torch
        self.device = device
        self.dataset = load_forward_dataset(cfg.dataset_path)
        self.k_path = recover_k_path(self.dataset)
        if not self.k_path.is_physical:
            raise ValueError(
                "formal inverse query surrogates require physical k coordinates; "
                "normalized index-query fallback is prohibited"
            )
        self.input_repr = str(dataset_cfg.get("input_repr", dataset_cfg.get("representation", "epsilon_grid")))
        self.model_name = checkpoint_model_name
        first = make_input(
            {"rho": self.dataset.rho[0]},
            DataRepresentationSpec(input_repr=self.input_repr, output_target="band_query"),
        )
        self.model = build_model(
            self.model_name,
            input_shape=(1, *tuple(int(v) for v in np.asarray(first).shape)),
            output_shape=(self.dataset.nb,),
            k_dim=self.k_path.dimension,
            **{key: value for key, value in params.items() if key in get_model_spec(self.model_name).defaults},
        ).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.checkpoint_path = checkpoint_path

    def _candidate_inputs(self, candidates: Any) -> Any:
        if bool(getattr(candidates, "requires_grad", False)):
            flat = candidates.reshape(int(candidates.shape[0]), -1)
            if self.input_repr in {"epsilon_grid", "image_epsilon"}:
                return epsilon_from_rho_torch(flat)
            if self.input_repr in {"rho_grid", "image_binary"}:
                return lift_rho_to_grid_torch(flat)
            return flat
        arr = candidates.detach().cpu().numpy().astype(np.float32)
        rows = [
            make_input(
                {"rho": row.reshape(-1)},
                DataRepresentationSpec(input_repr=self.input_repr, output_target="band_query"),
            )
            for row in arr.reshape(int(arr.shape[0]), -1)
        ]
        return self.torch.as_tensor(np.stack(rows, axis=0), dtype=self.torch.float32, device=self.device)

    def predict_tensor(self, candidates: Any) -> Any:
        structure = self._candidate_inputs(candidates)
        k_coordinates = self.torch.as_tensor(self.k_path.coordinates, dtype=self.torch.float32, device=self.device)
        preds: list[Any] = []
        context = self.torch.enable_grad() if bool(getattr(candidates, "requires_grad", False)) else self.torch.no_grad()
        with context:
            for k_coord in k_coordinates:
                repeated_k = k_coord.reshape(1, -1).expand(int(structure.shape[0]), -1)
                preds.append(self.model(structure, repeated_k))
        prediction = self.torch.stack(preds, dim=1)
        expected = (int(structure.shape[0]), int(self.dataset.nk), int(self.dataset.nb))
        if tuple(int(item) for item in prediction.shape) != expected:
            raise ValueError(
                "query surrogate sweep did not restore the full Wh contract: "
                f"expected={expected}, actual={tuple(int(item) for item in prediction.shape)}"
            )
        return prediction


class _ForwardFrozenSurrogate:
    """Frozen full-spectrum forward surrogate used for legacy inverse ranking."""

    def __init__(self, cfg: GenerativeInverseConfig, *, torch: Any, device: Any) -> None:
        surrogate_cfg = _as_mapping(cfg.surrogate)
        checkpoint_path = _resolve_forward_surrogate_checkpoint(cfg)
        if checkpoint_path is None:
            raise ValueError(
                "Formal inverse ranking requires surrogate.checkpoint_path, surrogate.run_dir, "
                "or a completed forward benchmark run for surrogate.kind=forward"
            )
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        resolved = _as_mapping(payload.get("resolved_config"))
        dataset_cfg = _as_mapping(resolved.get("dataset"))
        model_params = _as_mapping(payload.get("model_params"))

        self.cfg = cfg
        self.torch = torch
        self.device = device
        self.dataset = load_forward_dataset(cfg.dataset_path)
        self.input_repr = str(surrogate_cfg.get("input_repr") or dataset_cfg.get("input_repr") or "rho_wedge")
        self.output_target = str(dataset_cfg.get("output_target") or cfg.condition_target)
        grid_shape_value = surrogate_cfg.get("grid_shape", dataset_cfg.get("grid_shape"))
        grid_shape = None
        if isinstance(grid_shape_value, (list, tuple)) and len(grid_shape_value) == 2:
            grid_shape = (int(grid_shape_value[0]), int(grid_shape_value[1]))
        self.spec = DataRepresentationSpec(
            input_repr=self.input_repr,  # type: ignore[arg-type]
            output_target=self.output_target,  # type: ignore[arg-type]
            grid_shape=grid_shape,
            k_points=int(dataset_cfg["k_points"]) if dataset_cfg.get("k_points") is not None else None,
            n_bands=int(dataset_cfg["n_bands"]) if dataset_cfg.get("n_bands") is not None else None,
        )
        self.model_name = str(surrogate_cfg.get("model_name") or payload.get("model_name") or "res_mlp")
        self.model = build_model(
            self.model_name,
            input_shape=tuple(int(v) for v in payload["input_shape"]),
            output_shape=tuple(int(v) for v in payload["output_shape"]),
            **{key: value for key, value in model_params.items() if key in get_model_spec(self.model_name).defaults},
        ).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.checkpoint_path = checkpoint_path

    def _candidate_inputs(self, candidates: Any) -> Any:
        if bool(getattr(candidates, "requires_grad", False)):
            flat = candidates.reshape(int(candidates.shape[0]), -1)
            if self.input_repr in {"rho_wedge", "param_vector"}:
                return flat
            if self.input_repr in {"epsilon_grid", "image_epsilon"}:
                return epsilon_from_rho_torch(flat, grid_shape=self.spec.grid_shape)
            if self.input_repr in {"rho_grid", "image_binary"}:
                return lift_rho_to_grid_torch(flat, grid_shape=self.spec.grid_shape)
            return flat
        arr = candidates.detach().cpu().numpy().astype(np.float32)
        rows = []
        for item in arr:
            if self.cfg.output_repr == "rho_grid":
                row = {"rho_grid": np.asarray(item, dtype=np.float32)}
            else:
                row = {"rho": np.asarray(item, dtype=np.float32).reshape(-1)}
            rows.append(make_input(row, self.spec))
        return self.torch.as_tensor(np.stack(rows, axis=0), dtype=self.torch.float32, device=self.device)

    def predict_tensor(self, candidates: Any) -> Any:
        x = self._candidate_inputs(candidates)
        context = self.torch.enable_grad() if bool(getattr(candidates, "requires_grad", False)) else self.torch.no_grad()
        with context:
            pred = self.model(x)
        expected_size = int(self.dataset.nk) * int(self.dataset.nb)
        if int(pred.reshape(int(pred.shape[0]), -1).shape[1]) != expected_size:
            raise ValueError(
                "forward surrogate did not restore the full Wh contract: "
                f"expected flattened size={expected_size}, actual={int(pred.reshape(int(pred.shape[0]), -1).shape[1])}"
            )
        return pred.reshape(int(pred.shape[0]), int(self.dataset.nk), int(self.dataset.nb))


def _resolve_query_surrogate_checkpoint(cfg: GenerativeInverseConfig) -> Path | None:
    surrogate_cfg = _as_mapping(cfg.surrogate)
    configured = surrogate_cfg.get("checkpoint_path")
    if configured:
        return Path(str(configured))

    run_dir_value = surrogate_cfg.get("run_dir")
    run_dirs: list[Path] = []
    if run_dir_value:
        run_dirs.append(Path(str(run_dir_value)))
    else:
        runs_root = Path(str(surrogate_cfg.get("runs_dir", cfg.runs_dir)))
        experiment_id = str(surrogate_cfg.get("scaling_experiment_id", "query_data_scaling_v2_1000_full"))
        scale_n = int(surrogate_cfg.get("scale_n", 1000))
        if runs_root.exists():
            for path in runs_root.rglob("leaderboard.csv"):
                if path.parent.name.startswith(f"{experiment_id}_N{scale_n}"):
                    run_dirs.append(path.parent)
    if not run_dirs:
        return None
    child_run = max(run_dirs, key=lambda item: item.stat().st_mtime)
    leaderboard = _read_csv_rows(child_run / "leaderboard.csv")
    if not leaderboard:
        return None
    best = leaderboard[0]
    protocol = str(surrogate_cfg.get("protocol") or best.get("protocol") or "unseen_structure_all_k")
    model_name = str(surrogate_cfg.get("model_name") or best.get("model") or "fno2d_query")
    seed = int(surrogate_cfg.get("seed", cfg.seed))
    checkpoint = child_run / "cases" / f"{protocol}_{model_name}_seed{seed}.pt"
    if checkpoint.is_file():
        return checkpoint
    candidates = sorted((child_run / "cases").glob(f"{protocol}_{model_name}_seed*.pt"))
    return candidates[0] if candidates else None


def _resolve_forward_surrogate_checkpoint(cfg: GenerativeInverseConfig) -> Path | None:
    surrogate_cfg = _as_mapping(cfg.surrogate)
    configured = surrogate_cfg.get("checkpoint_path")
    if configured:
        return Path(str(configured))

    run_dir_value = surrogate_cfg.get("run_dir")
    run_dirs: list[Path] = []
    if run_dir_value:
        run_dirs.append(Path(str(run_dir_value)))
    else:
        runs_root = Path(str(surrogate_cfg.get("runs_dir", cfg.runs_dir)))
        experiment_id = str(surrogate_cfg.get("forward_experiment_id", ""))
        if runs_root.exists() and experiment_id:
            for path in runs_root.rglob("leaderboard.csv"):
                if path.parent.name.startswith(experiment_id):
                    run_dirs.append(path.parent)
            for path in runs_root.rglob("model.pt"):
                if path.parent.name.startswith(experiment_id):
                    run_dirs.append(path.parent)
    if not run_dirs:
        return None

    run_dir = max(run_dirs, key=lambda item: item.stat().st_mtime)
    direct_checkpoint = run_dir / "model.pt"
    if direct_checkpoint.is_file():
        return direct_checkpoint

    leaderboard = _read_csv_rows(run_dir / "leaderboard.csv")
    if leaderboard:
        model_name = str(surrogate_cfg.get("model_name", "")).strip()
        input_repr = str(surrogate_cfg.get("input_repr", "")).strip()
        seed = str(surrogate_cfg.get("seed", "")).strip()
        candidates = [row for row in leaderboard if str(row.get("status", "")).lower() == "ok"]
        if model_name:
            candidates = [row for row in candidates if str(row.get("model", "")) == model_name]
        if input_repr:
            candidates = [row for row in candidates if str(row.get("input_repr", "")) == input_repr]
        if seed:
            candidates = [row for row in candidates if str(row.get("seed", "")) == seed]
        metric = str(surrogate_cfg.get("selection_metric", "avg_MRE_bands_2_10_pct"))
        if not candidates:
            candidates = [row for row in leaderboard if str(row.get("status", "")).lower() == "ok"]
        if candidates:
            def metric_value(row: Mapping[str, Any]) -> float:
                value = row.get(metric, row.get("mse", "inf"))
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return float("inf")

            best = min(candidates, key=metric_value)
            checkpoint = best.get("checkpoint_path")
            if checkpoint:
                path = Path(str(checkpoint))
                if path.is_file():
                    return path
            train_run = best.get("train_run") or best.get("train_run_dir")
            if train_run:
                path = Path(str(train_run)) / "model.pt"
                if path.is_file():
                    return path

    case_id = str(surrogate_cfg.get("case_id", "")).strip()
    if case_id:
        path = run_dir / "cases" / case_id / "train" / "model.pt"
        if path.is_file():
            return path
    case_checkpoints = sorted((run_dir / "cases").glob("*/train/model.pt"))
    return case_checkpoints[0] if case_checkpoints else None
