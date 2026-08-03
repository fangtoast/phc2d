"""Inverse-candidate MPB replay runtime.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.artifacts import save_inverse_replay_band_arrays
from phc2d.data import load_forward_dataset
from phc2d.data.k_path import build_square_gxm_reduced_spec, hash_k_path_points
from phc2d.inverse.config import load_inverse_run_config
from phc2d.plot import plot_metric_report
from phc2d.runs import (
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)
from phc2d.solvers.mpb import MpbBandSolverConfig, solve_bands_for_rho_mpb

BandSolver = Callable[[np.ndarray, list[list[float]], MpbBandSolverConfig], tuple[np.ndarray, dict[str, Any]]]


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _coerce_sequence(value: Any, *, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return list(default)


def _derive_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join(str(value) for value in (base_seed, *parts))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _relative_mre(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if pred.size == 0:
        return np.array([], dtype=np.float64)
    denom = np.maximum(np.abs(tgt), 1.0e-8)
    return np.abs(pred - tgt) / denom


def _mae(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64))


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0


def _proxy_predict(candidates: np.ndarray, target: np.ndarray, *, seed: int) -> np.ndarray:
    flat = np.asarray(candidates, dtype=np.float32).reshape(int(candidates.shape[0]), -1)
    target_size = int(np.prod(target.shape))
    if flat.size == 0:
        return np.zeros((0, target_size), dtype=np.float32).reshape(0, *target.shape)
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(int(flat.shape[1]), target_size), loc=0.0, scale=0.5).astype(np.float32)
    pred = flat @ weights
    return pred.reshape(flat.shape[0], *target.shape)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return str(value)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(_to_jsonable(dict(row)))


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(dict(row)), ensure_ascii=False, sort_keys=True) + "\n")


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _parse_yaml_text(text: str) -> dict[str, Any]:
    if not str(text).strip():
        return {}
    payload = yaml.safe_load(text) or {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _merge_mapping(primary: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(fallback)
    merged.update({key: value for key, value in primary.items() if value is not None})
    return merged


def _band_range_from_config(value: Any, *, nb: int, default: tuple[int, int]) -> tuple[int, int]:
    raw = _coerce_sequence(value, default=list(default))
    if len(raw) != 2:
        raise ValueError(f"Band range must contain [start_one_based, end_one_based], got {value!r}")
    start = int(raw[0])
    end = int(raw[1])
    if start < 1 or end < start or end > int(nb):
        raise ValueError(f"Invalid one-based band range [{start}, {end}] for Nb={nb}")
    return start, end


def _band_slice(band_range: tuple[int, int]) -> slice:
    start, end = band_range
    return slice(start - 1, end)


def _matrix_from_condition(raw: Any, *, target_shape: tuple[int, int], target_id: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.shape == target_shape:
        return arr
    if arr.size == int(np.prod(target_shape)):
        return arr.reshape(target_shape)
    raise ValueError(f"Target {target_id} condition cannot be reshaped to Wh shape {target_shape}: {arr.shape}")


def _select_candidates(target_record: Mapping[str, Any], *, top_m: int) -> tuple[np.ndarray, list[int]]:
    raw = np.asarray(target_record.get("candidates", []), dtype=np.float32)
    raw_fallback = np.asarray(target_record.get("raw_candidates", []), dtype=np.float32)
    if raw.ndim == 1 and raw.size > 0:
        raw = raw.reshape(1, -1)
    if raw_fallback.ndim == 1 and raw_fallback.size > 0:
        raw_fallback = raw_fallback.reshape(1, -1)
    if raw_fallback.size and (raw.size == 0 or int(raw.shape[0]) < int(top_m)):
        raw = raw_fallback
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.float32), []
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    target_count = min(max(int(top_m), 1), int(raw.shape[0]))
    selected = [int(item) for item in _coerce_sequence(target_record.get("selected_indices"), default=[])]
    ranked = [int(item) for item in _coerce_sequence(target_record.get("ranked_indices"), default=[])]
    selected = [item for item in selected if 0 <= item < int(raw.shape[0])]
    for item in ranked:
        if len(selected) >= target_count:
            break
        if 0 <= item < int(raw.shape[0]) and item not in selected:
            selected.append(item)
    for item in range(int(raw.shape[0])):
        if len(selected) >= target_count:
            break
        if item not in selected:
            selected.append(item)
    selected = selected[:target_count]
    return raw[np.asarray(selected, dtype=np.int64)], selected


def _candidate_metric(record: Mapping[str, Any], *, policy: str = "best_surrogate_mre_bands_2_10") -> float | None:
    metrics = _as_mapping(record.get("surrogate_metrics"))
    normalized = str(policy).strip().lower()
    if normalized in {"best_surrogate_mre_bands_2_10", "best_of_k_surrogate_mre_bands_2_10"}:
        keys = ("best_of_k_surrogate_mre_bands_2_10", "best_of_k_surrogate_mre", "best_of_k_surrogate_mre_full")
    elif normalized in {"best_surrogate_mre", "best_of_k_surrogate_mre"}:
        keys = ("best_of_k_surrogate_mre", "best_of_k_surrogate_mre_bands_2_10", "best_of_k_surrogate_mre_full")
    elif normalized in {"best_surrogate_mre_full", "best_of_k_surrogate_mre_full"}:
        keys = ("best_of_k_surrogate_mre_full", "best_of_k_surrogate_mre_bands_2_10", "best_of_k_surrogate_mre")
    else:
        raise ValueError(f"Unsupported within_target_record_policy: {policy!r}")
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _target_index(record: Mapping[str, Any], fallback_index: int) -> int:
    return int(record.get("target_index", fallback_index))


def _dedupe_target_records(
    targets: list[Mapping[str, Any]],
    *,
    within_target_record_policy: str,
) -> list[tuple[int, Mapping[str, Any]]]:
    chosen: dict[int, tuple[int, float | None, Mapping[str, Any]]] = {}
    for source_index, record in enumerate(targets):
        target_index = _target_index(record, source_index)
        metric = _candidate_metric(record, policy=within_target_record_policy)
        if target_index not in chosen:
            chosen[target_index] = (source_index, metric, record)
            continue
        old_source, old_metric, old_record = chosen[target_index]
        if metric is not None and math.isfinite(metric) and (old_metric is None or not math.isfinite(old_metric) or metric < old_metric):
            chosen[target_index] = (source_index, metric, record)
        else:
            chosen[target_index] = (old_source, old_metric, old_record)
    return [(source_index, record) for _target_index_value, (source_index, _metric, record) in sorted(chosen.items())]


def _load_target_indices(path: str | Path) -> list[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [int(item) for item in payload]
    if isinstance(payload, Mapping):
        values = payload.get("target_indices")
        if isinstance(values, list):
            return [int(item) for item in values]
    raise ValueError(f"target_indices_path must contain a list or mapping with target_indices: {path}")


def _write_target_indices(
    path: str | Path,
    *,
    selected: list[tuple[int, str, Mapping[str, Any]]],
    selection_unit: str,
    within_target_record_policy: str,
    candidates_path: Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    target_indices = [_target_index(record, source_index) for source_index, _stratum, record in selected]
    out.write_text(
        json.dumps(
            {
                "version": 1,
                "selection_unit": str(selection_unit),
                "within_target_record_policy": str(within_target_record_policy),
                "candidates_path": str(candidates_path),
                "target_indices": target_indices,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _select_target_records(
    targets: list[Mapping[str, Any]],
    *,
    n_targets: int,
    seed: int,
    selection_unit: str = "record",
    within_target_record_policy: str = "best_surrogate_mre_bands_2_10",
    target_indices: list[int] | None = None,
) -> list[tuple[int, str, Mapping[str, Any]]]:
    indexed_targets: list[tuple[int, Mapping[str, Any]]]
    normalized_unit = str(selection_unit).strip().lower()
    if normalized_unit in {"record", "target_record"}:
        indexed_targets = list(enumerate(targets))
    elif normalized_unit == "target_index":
        indexed_targets = _dedupe_target_records(targets, within_target_record_policy=within_target_record_policy)
    else:
        raise ValueError("replay.selection_unit must be either 'record' or 'target_index'")

    if target_indices is not None:
        by_target_index = {
            _target_index(record, source_index): (source_index, record)
            for source_index, record in indexed_targets
        }
        selected_from_path: list[tuple[int, str, Mapping[str, Any]]] = []
        requested_indices = [int(item) for item in target_indices][: max(1, int(n_targets))]
        for target_index_value in requested_indices:
            if target_index_value not in by_target_index:
                raise ValueError(f"target_indices_path requested missing target_index={target_index_value}")
            source_index, record = by_target_index[target_index_value]
            selected_from_path.append((source_index, "target_indices_path", record))
        return selected_from_path

    targets_for_selection = indexed_targets
    requested = max(1, min(int(n_targets), len(targets_for_selection)))
    scored: list[tuple[int, float, Mapping[str, Any]]] = []
    unscored: list[tuple[int, Mapping[str, Any]]] = []
    for index, record in targets_for_selection:
        metric = _candidate_metric(record, policy=within_target_record_policy)
        if metric is None or not math.isfinite(metric):
            unscored.append((index, record))
        else:
            scored.append((index, metric, record))

    if requested >= len(targets_for_selection) or not scored:
        return [(index, "sequential", record) for index, record in targets_for_selection[:requested]]

    scored.sort(key=lambda item: item[1])
    rng = np.random.default_rng(int(seed))
    quota = max(1, requested // 4)
    selected: list[tuple[int, str, Mapping[str, Any]]] = []
    used: set[int] = set()

    def add(rows: list[tuple[int, float, Mapping[str, Any]]], label: str, limit: int) -> None:
        for index, _metric, record in rows:
            if index in used:
                continue
            selected.append((index, label, record))
            used.add(index)
            if sum(1 for _idx, item_label, _row in selected if item_label == label) >= limit:
                break

    add(scored, "low_surrogate_mre", quota)
    middle_start = max(0, len(scored) // 2 - quota // 2)
    add(scored[middle_start:] + scored[:middle_start], "median_surrogate_mre", quota)
    add(list(reversed(scored)), "high_surrogate_mre", quota)

    remaining = [row for row in scored if row[0] not in used]
    if remaining:
        order = rng.permutation(len(remaining)).tolist()
        for pos in order:
            index, _metric, record = remaining[pos]
            if len(selected) >= requested:
                break
            selected.append((index, "fixed_random", record))
            used.add(index)
    for index, record in unscored:
        if len(selected) >= requested:
            break
        if index not in used:
            selected.append((index, "unscored_fill", record))
            used.add(index)

    return selected[:requested]


def _resolve_candidates_path(cfg, *, replay_cfg: Mapping[str, Any], output_cfg: Mapping[str, Any]) -> Path:
    explicit = str(replay_cfg.get("candidates_path") or replay_cfg.get("strategy_candidates_path") or "").strip()
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"candidates_path not found: {path}")
        return path

    if cfg.run_dir is not None:
        run_dir = Path(cfg.run_dir)
        candidate_path = run_dir / "candidates.json"
        if candidate_path.exists():
            return candidate_path

    runs_root = Path(output_cfg.get("runs_dir", cfg.output_dir))
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    source_experiment_id = str(replay_cfg.get("source_experiment_id", cfg.experiment_id)).strip()
    candidates = []
    for filename in ("strategy_candidates.json", "candidates.json"):
        candidates.extend(
            path
            for path in runs_root.rglob(filename)
            if path.parent.name.startswith(source_experiment_id)
        )
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-resolve candidate payload; set replay.candidates_path or run inverse sample/rerank first"
        )
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _load_candidate_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise TypeError(f"Candidate payload must be a mapping: {path}")
    return payload


def _resolve_k_points(raw: Mapping[str, Any], dataset_config: Mapping[str, Any], *, nk: int) -> tuple[list[list[float]], dict[str, Any]]:
    replay_k_cfg = _as_mapping(raw.get("k_path"))
    dataset_k_cfg = _as_mapping(dataset_config.get("k_path"))
    k_cfg = _merge_mapping(replay_k_cfg, dataset_k_cfg)
    for key in ("points", "coordinates", "k_points"):
        if k_cfg.get(key) is None:
            continue
        points = np.asarray(k_cfg[key], dtype=np.float64)
        if points.ndim != 2 or points.shape[0] != int(nk):
            raise ValueError(f"k_path.{key} must have shape (Nk, dim), got {points.shape} for Nk={nk}")
        return points.astype(float).tolist(), {
            "source": f"config.k_path.{key}",
            "k_points_hash": hash_k_path_points(points),
            "coordinate_system": str(k_cfg.get("coordinate_system", "")),
        }

    family = str(k_cfg.get("family", "square_gxm")).strip().lower()
    n_total = _coerce_int(k_cfg.get("n_total"), default=nk)
    if family not in {"square_gxm", "gxm", "square_gxm_reduced"}:
        raise ValueError(f"Cannot reconstruct k_path family {family!r}; provide explicit k_points.")
    spec = build_square_gxm_reduced_spec(
        n_total=n_total,
        lattice_constant=_coerce_float(_as_mapping(dataset_config.get("solver")).get("lattice_constant"), 1.0),
        k_path_id=str(k_cfg.get("k_path_id", "inverse_replay_square_gxm_reduced_v1")),
    )
    points = np.asarray(spec.k_points, dtype=np.float64)
    if points.shape[0] != int(nk):
        raise ValueError(f"Reconstructed k_path Nk={points.shape[0]} does not match dataset Nk={nk}")
    return points.astype(float).tolist(), {
        "source": "reconstructed_square_gxm_reduced_spec",
        "k_path_id": spec.k_path_id,
        "k_points_hash": spec.k_points_hash,
        "coordinate_system": spec.coordinate_system,
    }


def _build_solver_config(raw: Mapping[str, Any], dataset_config: Mapping[str, Any], *, nb: int) -> MpbBandSolverConfig:
    solver_cfg = _merge_mapping(_as_mapping(raw.get("solver")), _as_mapping(dataset_config.get("solver")))
    geometry_cfg = _merge_mapping(_as_mapping(raw.get("geometry")), _as_mapping(dataset_config.get("geometry")))
    grid_shape = geometry_cfg.get("grid_shape")
    if grid_shape is not None:
        grid_shape = tuple(int(v) for v in _coerce_sequence(grid_shape, default=[]))
    return MpbBandSolverConfig(
        num_bands=int(nb),
        resolution=_coerce_int(solver_cfg.get("resolution"), 64),
        polarization=str(solver_cfg.get("polarization", "TE")),
        epsilon_air=_coerce_float(solver_cfg.get("epsilon_air"), 1.0),
        epsilon_material=_coerce_float(solver_cfg.get("epsilon_material"), 8.9),
        lattice_constant=_coerce_float(solver_cfg.get("lattice_constant"), 1.0),
        eigensolver_tolerance=_coerce_float(solver_cfg.get("eigensolver_tolerance"), 1.0e-7),
        grid_shape=grid_shape,  # type: ignore[arg-type]
        band_guard_count=_coerce_int(solver_cfg.get("band_guard_count"), 0),
        band_order_policy=str(solver_cfg.get("band_order_policy", "frequency_ascending")),
    )


def _fake_solve_bands_for_rho(
    rho_wedge: np.ndarray,
    k_points: list[list[float]],
    config: MpbBandSolverConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    rho = np.asarray(rho_wedge, dtype=np.float64).reshape(-1)
    fill = float(np.mean(rho >= 0.5)) if rho.size else 0.0
    k = np.asarray(k_points, dtype=np.float64)
    k_norm = np.linalg.norm(k, axis=1) if k.ndim == 2 else np.arange(len(k_points), dtype=np.float64)
    base = 0.05 + 0.01 * np.arange(int(config.num_bands), dtype=np.float64)
    bands = base[None, :] + 0.02 * k_norm[:, None] + 0.01 * fill
    return bands.astype(np.float32), {
        "mpb_backend": "test_fake_solver",
        "occupied_fill_fraction": fill,
        "num_bands": int(config.num_bands),
    }


def _run_candidate_replay(
    candidates: np.ndarray,
    target_matrix: np.ndarray,
    *,
    backend: str,
    seed: int,
    k_points: list[list[float]],
    solver_config: MpbBandSolverConfig,
    solver_fn: BandSolver | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if backend == "proxy":
        surrogate_pred = _proxy_predict(candidates, target_matrix, seed=seed)
        rng = np.random.default_rng(_derive_seed(seed, "proxy_replay"))
        replay_pred = surrogate_pred + 0.05 * rng.normal(size=surrogate_pred.shape)
        return replay_pred.astype(np.float32), [{"mpb_backend": "proxy_not_physical"} for _ in range(int(candidates.shape[0]))]

    if backend not in {"fake_mpb", "real_mpb"}:
        raise ValueError(f"Unsupported replay backend: {backend!r}")

    predictions: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for candidate in np.asarray(candidates, dtype=np.float32):
        if solver_fn is not None:
            bands, meta = solver_fn(candidate, k_points, solver_config)
        elif backend == "fake_mpb":
            bands, meta = _fake_solve_bands_for_rho(candidate, k_points, solver_config)
        else:
            bands, meta = solve_bands_for_rho_mpb(candidate, k_points=k_points, config=solver_config)
        arr = np.asarray(bands, dtype=np.float32)
        if arr.shape != target_matrix.shape:
            raise ValueError(f"Replay solver returned bands shape {arr.shape}, expected {target_matrix.shape}")
        predictions.append(arr)
        metadata.append(dict(meta))
    return np.stack(predictions, axis=0), metadata


def _extract_surrogate_predictions(
    target_record: Mapping[str, Any],
    *,
    selected_indices: list[int],
    target_shape: tuple[int, int],
) -> tuple[np.ndarray | None, str]:
    for key in ("selected_surrogate_predictions", "surrogate_predictions", "surrogate_Wh", "surrogate_pred"):
        raw = target_record.get(key)
        if raw is None:
            raw = _as_mapping(target_record.get("surrogate_metrics")).get(key)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape == target_shape:
            arr = arr.reshape(1, *target_shape)
        if arr.ndim == 3 and arr.shape[1:] == target_shape:
            if arr.shape[0] == len(selected_indices):
                return arr, key
            if arr.shape[0] > max(selected_indices):
                return arr[np.asarray(selected_indices, dtype=np.int64)], f"{key}[selected_indices]"
    return None, "unavailable"


def _best_error(predictions: np.ndarray, target: np.ndarray, band_range: tuple[int, int]) -> tuple[float, int]:
    band_sel = _band_slice(band_range)
    errors = _relative_mre(predictions[:, :, band_sel], target[:, band_sel]).reshape(int(predictions.shape[0]), -1).mean(axis=1)
    best_idx = int(np.argmin(errors)) if errors.size else 0
    return float(errors[best_idx]), best_idx


def _scalar_surrogate_mre(target_record: Mapping[str, Any], key: str) -> float | None:
    metrics = _as_mapping(target_record.get("surrogate_metrics"))
    value = metrics.get(key)
    if value is None and key == "best_of_k_surrogate_mre_bands_1_10":
        value = metrics.get("best_of_k_surrogate_mre_full")
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _complete_gap_labels(wh: np.ndarray, *, threshold_midgap_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(wh, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float64)
    labels: list[bool] = []
    ratios: list[float] = []
    for band in range(int(matrix.shape[1]) - 1):
        lower_top = float(np.max(matrix[:, band]))
        upper_bottom = float(np.min(matrix[:, band + 1]))
        width = upper_bottom - lower_top
        center = 0.5 * (upper_bottom + lower_top)
        ratio = width / max(abs(center), 1.0e-8)
        labels.append(bool(ratio > float(threshold_midgap_ratio)))
        ratios.append(float(ratio))
    return np.asarray(labels, dtype=bool), np.asarray(ratios, dtype=np.float64)


def _classification_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(true & pred))
    fp = int(np.sum(~true & pred))
    fn = int(np.sum(true & ~pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "gap_precision": float(precision),
        "gap_recall": float(recall),
        "gap_f1": float(f1),
        "gap_true_positive": float(tp),
        "gap_false_positive": float(fp),
        "gap_false_negative": float(fn),
        "gap_label_count": float(len(true)),
    }


def _aggregate_per_band(best_rows: list[dict[str, Any]], *, nb: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band_idx in range(int(nb)):
        band_rows = [row for row in best_rows if int(row["band_one_based"]) == band_idx + 1]
        replay_mre = [float(row["mpb_replay_mre_pct"]) for row in band_rows]
        surrogate_mre = [
            float(row["surrogate_mre_pct"])
            for row in band_rows
            if row.get("surrogate_mre_pct") is not None and math.isfinite(float(row["surrogate_mre_pct"]))
        ]
        surrogate_sources = sorted({str(row.get("surrogate_mre_source", "")) for row in band_rows if row.get("surrogate_mre_pct") is not None})
        rows.append(
            {
                "band_one_based": band_idx + 1,
                "sample_count": len(band_rows),
                "mpb_replay_mre_pct": float(np.mean(replay_mre)) if replay_mre else 0.0,
                "surrogate_mre_pct": float(np.mean(surrogate_mre)) if surrogate_mre else None,
                "surrogate_mre_source": "+".join(surrogate_sources) if surrogate_sources else "unavailable",
                "surrogate_replay_gap_pct": None
                if not surrogate_mre or not replay_mre
                else float(np.mean(replay_mre) - np.mean(surrogate_mre)),
                "mae": float(np.mean([float(row["mae"]) for row in band_rows])) if band_rows else 0.0,
                "rmse": float(np.mean([float(row["rmse"]) for row in band_rows])) if band_rows else 0.0,
            }
        )
    return rows


def _extract_strategy_surrogate_predictions(
    ranking: Mapping[str, Any],
    *,
    selected_indices: list[int],
    target_shape: tuple[int, int],
) -> tuple[np.ndarray | None, str]:
    for key in ("selected_surrogate_predictions", "surrogate_predictions", "surrogate_Wh", "surrogate_pred"):
        raw = ranking.get(key)
        if raw is None:
            raw = _as_mapping(ranking.get("surrogate_metrics")).get(key)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape == target_shape:
            arr = arr.reshape(1, *target_shape)
        if arr.ndim == 3 and arr.shape[1:] == target_shape:
            if arr.shape[0] == len(selected_indices):
                return arr, key
            if selected_indices and arr.shape[0] > max(selected_indices):
                return arr[np.asarray(selected_indices, dtype=np.int64)], f"{key}[selected_indices]"
    return None, "unavailable"


def _strategy_selected_indices(ranking: Mapping[str, Any], *, candidate_count: int, top_m: int) -> list[int]:
    selected = [int(item) for item in _coerce_sequence(ranking.get("selected_indices"), default=[])]
    ranked = [int(item) for item in _coerce_sequence(ranking.get("ranked_indices"), default=[])]
    selected = [item for item in selected if 0 <= item < int(candidate_count)]
    target_count = min(max(int(top_m), 1), int(candidate_count))
    for item in ranked:
        if len(selected) >= target_count:
            break
        if 0 <= item < int(candidate_count) and item not in selected:
            selected.append(item)
    for item in range(int(candidate_count)):
        if len(selected) >= target_count:
            break
        if item not in selected:
            selected.append(item)
    return selected[:target_count]


def _aggregate_per_strategy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strategy_ids = sorted({str(row.get("strategy_id", "")) for row in rows})
    output: list[dict[str, Any]] = []

    def mean_key(strategy_rows: list[dict[str, Any]], key: str) -> float | None:
        numbers: list[float] = []
        for row in strategy_rows:
            value = row.get(key)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                numbers.append(number)
        return float(np.mean(numbers)) if numbers else None

    for strategy_id in strategy_ids:
        strategy_rows = [row for row in rows if str(row.get("strategy_id", "")) == strategy_id]
        output.append(
            {
                "strategy_id": strategy_id,
                "target_count": int(len(strategy_rows)),
                "best_of_k_replay_mre_bands_2_10_pct": mean_key(
                    strategy_rows, "best_of_k_replay_mre_bands_2_10_pct"
                ),
                "best_of_k_replay_mre_bands_1_10_pct": mean_key(
                    strategy_rows, "best_of_k_replay_mre_bands_1_10_pct"
                ),
                "best_of_k_surrogate_mre_bands_2_10_pct": mean_key(
                    strategy_rows, "best_of_k_surrogate_mre_bands_2_10_pct"
                ),
                "best_of_k_surrogate_mre_bands_1_10_pct": mean_key(
                    strategy_rows, "best_of_k_surrogate_mre_bands_1_10_pct"
                ),
                "surrogate_replay_gap_bands_2_10_pct": mean_key(
                    strategy_rows, "surrogate_replay_gap_bands_2_10_pct"
                ),
                "surrogate_replay_gap_bands_1_10_pct": mean_key(
                    strategy_rows, "surrogate_replay_gap_bands_1_10_pct"
                ),
                "mpb_replay_time_s": mean_key(strategy_rows, "mpb_replay_time_s"),
                "candidate_count_selected": mean_key(strategy_rows, "candidate_count_selected"),
            }
        )
    return output


def _aggregate_per_strategy_band(best_rows: list[dict[str, Any]], *, nb: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy_id in sorted({str(row.get("strategy_id", "")) for row in best_rows}):
        rows = [row for row in best_rows if str(row.get("strategy_id", "")) == strategy_id]
        for row in _aggregate_per_band(rows, nb=nb):
            row["strategy_id"] = strategy_id
            output.append(row)
    return output


def _save_selected_band_artifacts(
    run_dir: Path,
    *,
    target_bands: list[np.ndarray],
    surrogate_pred_bands: list[np.ndarray | None],
    mpb_replay_bands: list[np.ndarray],
    surrogate_sources: list[str],
) -> dict[str, Any]:
    if not target_bands:
        raise ValueError("No selected replay band arrays were collected")
    target_array = np.stack([np.asarray(value, dtype=np.float32) for value in target_bands], axis=0)
    replay_array = np.stack([np.asarray(value, dtype=np.float32) for value in mpb_replay_bands], axis=0)
    surrogate_array = np.full_like(target_array, np.nan, dtype=np.float32)
    surrogate_available_count = 0
    for index, value in enumerate(surrogate_pred_bands):
        if value is not None:
            surrogate_array[index] = np.asarray(value, dtype=np.float32)
            surrogate_available_count += 1
    return save_inverse_replay_band_arrays(
        run_dir,
        target_bands=target_array,
        surrogate_pred_bands=surrogate_array,
        mpb_replay_bands=replay_array,
        surrogate_available_count=surrogate_available_count,
        surrogate_sources=surrogate_sources,
    )


def _run_strategy_replay_payload(
    *,
    cfg: Any,
    run_dir: Path,
    candidates_path: Path,
    target_records: list[tuple[int, str, Mapping[str, Any]]],
    target_shape: tuple[int, int],
    nb: int,
    replay_top_m: int,
    backend: str,
    k_points: list[list[float]],
    solver_config: MpbBandSolverConfig,
    solver_fn: BandSolver | None,
    band_range_2_10: tuple[int, int],
    band_range_1_10: tuple[int, int],
    primary_range: tuple[int, int],
    gap_threshold: float,
    validation_boundary: str,
    timeout_seconds: float,
    wall_start: float,
    n_targets_cfg: int,
    replay_cfg: Mapping[str, Any],
    k_path_meta: Mapping[str, Any],
) -> dict[str, Any]:
    target_rows: list[dict[str, Any]] = []
    candidate_jsonl_rows: list[dict[str, Any]] = []
    best_band_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    gap_true: list[bool] = []
    gap_pred: list[bool] = []
    success_count = 0
    unique_candidate_solve_count = 0
    strategy_candidate_selection_count = 0
    target_elapsed_times: list[float] = []
    selected_target_bands: list[np.ndarray] = []
    selected_surrogate_bands: list[np.ndarray | None] = []
    selected_replay_bands: list[np.ndarray] = []
    selected_surrogate_sources: list[str] = []

    for local_pos, (source_index, stratum, target_record) in enumerate(target_records):
        start = time.perf_counter()
        target_id = str(target_record.get("target_id", f"target_{source_index:06d}"))
        condition = _as_mapping(target_record.get("condition"))
        target_matrix = _matrix_from_condition(
            condition.get(cfg.condition_target, condition.get("target", [])),
            target_shape=target_shape,
            target_id=target_id,
        )
        raw_candidates = np.asarray(target_record.get("candidates", target_record.get("raw_candidates", [])), dtype=np.float32)
        if raw_candidates.ndim == 1 and raw_candidates.size:
            raw_candidates = raw_candidates.reshape(1, -1)
        if raw_candidates.size == 0:
            raise ValueError(f"Target {target_id} has no strategy candidate pool")

        rankings = _as_mapping(target_record.get("strategy_rankings"))
        if not rankings:
            raise ValueError(f"Target {target_id} has no strategy_rankings")
        selected_by_strategy: dict[str, list[int]] = {}
        union_indices: list[int] = []
        for strategy_id, raw_ranking in rankings.items():
            ranking = _as_mapping(raw_ranking)
            selected = _strategy_selected_indices(
                ranking,
                candidate_count=int(raw_candidates.shape[0]),
                top_m=int(replay_top_m),
            )
            selected_by_strategy[str(strategy_id)] = selected
            for index in selected:
                if index not in union_indices:
                    union_indices.append(index)
        if not union_indices:
            raise ValueError(f"Target {target_id} has no selected strategy candidates")

        candidates = raw_candidates[np.asarray(union_indices, dtype=np.int64)]
        replay_pred, solver_metadata = _run_candidate_replay(
            candidates,
            target_matrix,
            backend=backend,
            seed=_derive_seed(cfg.seed, target_id, local_pos),
            k_points=k_points,
            solver_config=solver_config,
            solver_fn=solver_fn,
        )
        elapsed = float(time.perf_counter() - start)
        target_elapsed_times.append(elapsed)
        success_count += 1
        unique_candidate_solve_count += int(len(union_indices))
        strategy_candidate_selection_count += int(sum(len(value) for value in selected_by_strategy.values()))
        union_position = {int(index): pos for pos, index in enumerate(union_indices)}

        for strategy_id, selected_indices in selected_by_strategy.items():
            ranking = _as_mapping(rankings.get(strategy_id))
            positions = [union_position[int(index)] for index in selected_indices]
            strategy_replay = replay_pred[np.asarray(positions, dtype=np.int64)]
            surrogate_pred, surrogate_source = _extract_strategy_surrogate_predictions(
                ranking,
                selected_indices=selected_indices,
                target_shape=target_shape,
            )
            replay_2_10, best_idx = _best_error(strategy_replay, target_matrix, band_range_2_10)
            replay_1_10, _ = _best_error(strategy_replay, target_matrix, band_range_1_10)
            replay_primary, best_primary_idx = _best_error(strategy_replay, target_matrix, primary_range)
            if best_primary_idx != best_idx:
                best_idx = best_primary_idx
            if surrogate_pred is not None:
                surrogate_2_10, _ = _best_error(surrogate_pred, target_matrix, band_range_2_10)
                surrogate_1_10, _ = _best_error(surrogate_pred, target_matrix, band_range_1_10)
            else:
                scalar_record = {"surrogate_metrics": _as_mapping(ranking.get("surrogate_metrics"))}
                surrogate_2_10 = _scalar_surrogate_mre(scalar_record, "best_of_k_surrogate_mre_bands_2_10")
                surrogate_1_10 = _scalar_surrogate_mre(scalar_record, "best_of_k_surrogate_mre_bands_1_10")

            best_replay = strategy_replay[best_idx]
            best_surrogate = (
                surrogate_pred[best_idx]
                if surrogate_pred is not None and best_idx < int(surrogate_pred.shape[0])
                else None
            )
            selected_target_bands.append(np.asarray(target_matrix, dtype=np.float32))
            selected_replay_bands.append(np.asarray(best_replay, dtype=np.float32))
            selected_surrogate_bands.append(
                np.asarray(best_surrogate, dtype=np.float32) if best_surrogate is not None else None
            )
            selected_surrogate_sources.append(str(surrogate_source))
            target_labels, target_ratios = _complete_gap_labels(target_matrix, threshold_midgap_ratio=gap_threshold)
            replay_labels, replay_ratios = _complete_gap_labels(best_replay, threshold_midgap_ratio=gap_threshold)
            for gap_idx in range(min(len(target_labels), len(replay_labels))):
                gap_true.append(bool(target_labels[gap_idx]))
                gap_pred.append(bool(replay_labels[gap_idx]))
                gap_rows.append(
                    {
                        "strategy_id": strategy_id,
                        "target_id": target_id,
                        "target_index": int(target_record.get("target_index", source_index)),
                        "gap_pair_lower_band_one_based": int(gap_idx + 1),
                        "gap_pair_upper_band_one_based": int(gap_idx + 2),
                        "target_gap_label": bool(target_labels[gap_idx]),
                        "replay_gap_label": bool(replay_labels[gap_idx]),
                        "target_gap_midgap_ratio": float(target_ratios[gap_idx]),
                        "replay_gap_midgap_ratio": float(replay_ratios[gap_idx]),
                    }
                )

            for band_idx in range(int(nb)):
                replay_band = best_replay[:, band_idx]
                target_band = target_matrix[:, band_idx]
                surrogate_band_mre: float | None = None
                surrogate_band_source = surrogate_source
                if best_surrogate is not None:
                    surrogate_band_mre = float(np.mean(_relative_mre(best_surrogate[:, band_idx], target_band)) * 100.0)
                    surrogate_band_source = "prediction"
                elif (band_idx + 1) >= band_range_2_10[0] and (band_idx + 1) <= band_range_2_10[1] and surrogate_2_10 is not None:
                    surrogate_band_mre = float(surrogate_2_10 * 100.0)
                    surrogate_band_source = "target_scalar_bands_2_10_broadcast"
                elif surrogate_1_10 is not None:
                    surrogate_band_mre = float(surrogate_1_10 * 100.0)
                    surrogate_band_source = "target_scalar_bands_1_10_broadcast"
                replay_band_mre = float(np.mean(_relative_mre(replay_band, target_band)) * 100.0)
                best_band_rows.append(
                    {
                        "strategy_id": strategy_id,
                        "target_id": target_id,
                        "target_index": int(target_record.get("target_index", source_index)),
                        "stratum": stratum,
                        "band_one_based": int(band_idx + 1),
                        "selected_candidate_index": int(selected_indices[best_idx]) if best_idx < len(selected_indices) else int(best_idx),
                        "surrogate_mre_pct": surrogate_band_mre,
                        "surrogate_mre_source": surrogate_band_source,
                        "mpb_replay_mre_pct": replay_band_mre,
                        "surrogate_replay_gap_pct": None if surrogate_band_mre is None else float(replay_band_mre - surrogate_band_mre),
                        "mae": float(np.mean(_mae(replay_band, target_band))),
                        "rmse": _rmse(replay_band, target_band),
                        "sample_count": int(target_band.shape[0]),
                    }
                )

            target_row = {
                "stage": "replay",
                "strategy_id": strategy_id,
                "target_id": target_id,
                "target_index": int(target_record.get("target_index", source_index)),
                "sample_id": str(target_record.get("sample_id", target_id)),
                "dataset_id": str(target_record.get("dataset_id", "")),
                "split": str(target_record.get("split", "unassigned")),
                "stratum": stratum,
                "candidate_count_selected": int(len(selected_indices)),
                "candidate_count_unique_for_target": int(len(union_indices)),
                "selected_indices": selected_indices,
                "best_candidate_selected_index": int(selected_indices[best_idx]) if best_idx < len(selected_indices) else int(best_idx),
                "best_of_k_replay_mre_bands_2_10_pct": float(replay_2_10 * 100.0),
                "best_of_k_replay_mre_bands_1_10_pct": float(replay_1_10 * 100.0),
                "best_of_k_replay_mre_primary_pct": float(replay_primary * 100.0),
                "best_of_k_surrogate_mre_bands_2_10_pct": None if surrogate_2_10 is None else float(surrogate_2_10 * 100.0),
                "best_of_k_surrogate_mre_bands_1_10_pct": None if surrogate_1_10 is None else float(surrogate_1_10 * 100.0),
                "surrogate_replay_gap_bands_2_10_pct": None if surrogate_2_10 is None else float(replay_2_10 * 100.0 - surrogate_2_10 * 100.0),
                "surrogate_replay_gap_bands_1_10_pct": None if surrogate_1_10 is None else float(replay_1_10 * 100.0 - surrogate_1_10 * 100.0),
                "mpb_replay_time_s": elapsed,
                "mpb_replay_time_per_candidate_s": elapsed / max(int(len(union_indices)), 1),
                "surrogate_mre_source": surrogate_source,
            }
            target_rows.append(target_row)
            for candidate_pos, selected in enumerate(selected_indices):
                union_pos = union_position[int(selected)]
                candidate_jsonl_rows.append(
                    {
                        "strategy_id": strategy_id,
                        "target_id": target_id,
                        "target_index": int(target_record.get("target_index", source_index)),
                        "candidate_local_pos": int(candidate_pos),
                        "selected_candidate_index": int(selected),
                        "dedup_union_position": int(union_pos),
                        "solver_metadata": solver_metadata[union_pos] if union_pos < len(solver_metadata) else {},
                    }
                )

    target_count = len(target_rows)
    replay_success_rate = float(success_count / max(len(target_records), 1))
    gap_metrics = _classification_metrics(gap_true, gap_pred)

    def mean_key(key: str) -> float | None:
        values = [row.get(key) for row in target_rows]
        numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        return float(np.mean(numbers)) if numbers else None

    total_time = float(sum(target_elapsed_times))
    metrics = {
        "best_of_k_replay_mre_bands_2_10_pct": mean_key("best_of_k_replay_mre_bands_2_10_pct"),
        "best_of_k_replay_mre_bands_1_10_pct": mean_key("best_of_k_replay_mre_bands_1_10_pct"),
        "best_of_k_surrogate_mre_bands_2_10_pct": mean_key("best_of_k_surrogate_mre_bands_2_10_pct"),
        "best_of_k_surrogate_mre_bands_1_10_pct": mean_key("best_of_k_surrogate_mre_bands_1_10_pct"),
        "surrogate_replay_gap_bands_2_10_pct": mean_key("surrogate_replay_gap_bands_2_10_pct"),
        "surrogate_replay_gap_bands_1_10_pct": mean_key("surrogate_replay_gap_bands_1_10_pct"),
        "mpb_replay_success_rate": replay_success_rate,
        "mpb_replay_time_s": total_time,
        "mpb_replay_time_per_candidate_s": total_time / max(int(unique_candidate_solve_count), 1),
        "wall_time_total_s": float(time.perf_counter() - wall_start),
        "targets": int(len(target_records)),
        "target_strategy_evaluations": int(target_count),
        "strategy_count": int(len({str(row.get("strategy_id", "")) for row in target_rows})),
        "strategy_candidate_selection_count": int(strategy_candidate_selection_count),
        "unique_candidate_solve_count": int(unique_candidate_solve_count),
        "candidate_dedupe_savings_count": int(strategy_candidate_selection_count - unique_candidate_solve_count),
        "candidate_dedupe_ratio": float(
            1.0 - unique_candidate_solve_count / max(strategy_candidate_selection_count, 1)
        ),
        "replay_timeout_seconds": timeout_seconds,
        "replay_top_m": int(replay_top_m),
        "replay_backend": backend,
        "validation_boundary": validation_boundary,
        "strategy_replay": True,
        **gap_metrics,
    }
    write_metrics(run_dir, metrics)

    per_band_rows = _aggregate_per_band(best_band_rows, nb=nb)
    per_strategy_rows = _aggregate_per_strategy(target_rows)
    per_strategy_band_rows = _aggregate_per_strategy_band(best_band_rows, nb=nb)
    _write_csv(run_dir / "history.csv", target_rows)
    _write_csv(run_dir / "per_target_replay_metrics.csv", target_rows)
    _write_csv(run_dir / "per_strategy_replay_metrics.csv", per_strategy_rows)
    _write_csv(run_dir / "per_band_replay_metrics.csv", per_band_rows)
    _write_csv(run_dir / "per_strategy_band_replay_metrics.csv", per_strategy_band_rows)
    _write_csv(run_dir / "per_target_band_replay_metrics.csv", best_band_rows)
    _write_csv(run_dir / "gap_metrics.csv", gap_rows)
    _write_jsonl(run_dir / "replay_candidates.jsonl", candidate_jsonl_rows)
    band_artifacts = _save_selected_band_artifacts(
        run_dir,
        target_bands=selected_target_bands,
        surrogate_pred_bands=selected_surrogate_bands,
        mpb_replay_bands=selected_replay_bands,
        surrogate_sources=selected_surrogate_sources,
    )

    write_results(
        run_dir,
        {
            "action": "replay mpb",
            "stage": "inverse_strategy_candidate_replay",
            "run_dir": str(run_dir),
            "experiment_id": cfg.experiment_id,
            "candidates_path": str(candidates_path),
            "dataset_path": str(cfg.dataset_path),
            "validation_boundary": validation_boundary,
            "replay_backend": backend,
            "strategy_replay": True,
            "band_ranges": {
                "bands_1_10": list(band_range_1_10),
                "bands_2_10": list(band_range_2_10),
                "primary": list(primary_range),
                "band_indexing": "one_based_inclusive",
            },
            "k_path": dict(k_path_meta),
            "target_shape": list(target_shape),
            "targets": target_rows,
            "per_band_metrics_path": "per_band_replay_metrics.csv",
            "per_strategy_metrics_path": "per_strategy_replay_metrics.csv",
            "per_strategy_band_metrics_path": "per_strategy_band_replay_metrics.csv",
            "per_target_metrics_path": "per_target_replay_metrics.csv",
            "gap_metrics_path": "gap_metrics.csv",
            "replay_candidates_path": "replay_candidates.jsonl",
            "band_array_artifacts": band_artifacts,
            "n_targets_requested": int(n_targets_cfg),
            "n_targets_processed": int(len(target_records)),
            "replay_enabled": bool(replay_cfg.get("enabled", True)),
        },
    )
    return metrics


def run_replay_mpb(config_path: str | Path, *, solver_fn: BandSolver | None = None) -> Path:
    cfg = load_inverse_run_config(config_path)
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Invalid replay config: {config_path}")

    replay_cfg = _as_mapping(raw.get("replay"))
    output_cfg = _as_mapping(raw.get("output"))
    metrics_cfg = _as_mapping(raw.get("metrics"))
    backend = str(replay_cfg.get("backend", "proxy")).strip().lower()
    if replay_cfg.get("require_real_mpb") and backend != "real_mpb":
        raise ValueError("replay.require_real_mpb=true requires replay.backend: real_mpb")

    bundle = load_forward_dataset(cfg.dataset_path)
    target_shape = (int(bundle.nk), int(bundle.nb))
    dataset_config = _parse_yaml_text(bundle.config_yaml)
    band_ranges_cfg = _as_mapping(metrics_cfg.get("band_ranges"))
    band_range_1_10 = _band_range_from_config(
        band_ranges_cfg.get("bands_1_10"),
        nb=bundle.nb,
        default=(1, bundle.nb),
    )
    band_range_2_10 = _band_range_from_config(
        band_ranges_cfg.get("bands_2_10"),
        nb=bundle.nb,
        default=(2 if bundle.nb >= 2 else 1, bundle.nb),
    )
    primary_range = _band_range_from_config(
        replay_cfg.get("primary_band_range"),
        nb=bundle.nb,
        default=band_range_2_10,
    )

    runs_root = Path(output_cfg.get("runs_dir", str(cfg.output_dir)))
    run_dir_override = output_cfg.get("run_dir")
    run_dir = Path(str(run_dir_override)) if run_dir_override else make_run_dir(cfg.experiment_id, runs_dir=runs_root)
    validation_boundary = {
        "real_mpb": "formal_inverse_candidate_mpb_replay",
        "fake_mpb": "test_fake_mpb_replay_not_physical",
        "proxy": "pilot_mpb_replay_proxy_not_physical",
    }.get(backend, "unknown_replay_boundary")
    if validation_boundary.startswith("formal") and solver_fn is not None:
        validation_boundary = "test_injected_solver_not_physical"

    logger = setup_logging(
        log_file=run_dir / "run.log",
        run_meta={
            "script": "phc2d replay mpb",
            "config_path": str(Path(config_path)),
            "validation_boundary": validation_boundary,
            "replay_backend": backend,
            "runs_root": str(runs_root),
        },
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_figures_dir(run_dir)
    write_resolved_config(run_dir, cfg.to_resolved_yaml())
    write_run_meta(
        run_dir,
        {
            "script": "phc2d replay mpb",
            "config_path": str(Path(config_path)),
            "validation_boundary": validation_boundary,
            "replay_backend": backend,
            "run_override": str(run_dir_override) if run_dir_override else "",
            "replay_enabled": bool(replay_cfg.get("enabled", True)),
            "environment": str(replay_cfg.get("environment", "WSL Ubuntu + conda meep_env" if backend == "real_mpb" else "local_test_or_proxy")),
        },
    )

    candidates_path = _resolve_candidates_path(cfg, replay_cfg=replay_cfg, output_cfg=output_cfg)
    payload = _load_candidate_payload(candidates_path)
    targets = payload.get("targets", []) if isinstance(payload, dict) else []
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"No replay targets found in {candidates_path}")

    replay_top_m = _coerce_int(replay_cfg.get("replay_top_m"), default=cfg.surrogate_top_m)
    n_targets_cfg = _coerce_int(replay_cfg.get("n_targets"), default=len(targets))
    selection_unit = str(replay_cfg.get("selection_unit", "record"))
    within_target_record_policy = str(replay_cfg.get("within_target_record_policy", "best_surrogate_mre_bands_2_10"))
    target_indices_path = str(replay_cfg.get("target_indices_path") or "").strip()
    target_indices = _load_target_indices(target_indices_path) if target_indices_path else None
    target_records = _select_target_records(
        targets,
        n_targets=n_targets_cfg,
        seed=_coerce_int(replay_cfg.get("selection_seed"), cfg.seed),
        selection_unit=selection_unit,
        within_target_record_policy=within_target_record_policy,
        target_indices=target_indices,
    )
    write_target_indices_path = str(replay_cfg.get("write_target_indices_path") or "").strip()
    if write_target_indices_path:
        _write_target_indices(
            write_target_indices_path,
            selected=target_records,
            selection_unit=selection_unit,
            within_target_record_policy=within_target_record_policy,
            candidates_path=candidates_path,
        )
    timeout_seconds = _coerce_float(replay_cfg.get("timeout_seconds"), 900.0)
    gap_cfg = _as_mapping(metrics_cfg.get("gap"))
    gap_threshold = _coerce_float(gap_cfg.get("threshold_midgap_ratio"), 0.0)

    k_points, k_path_meta = _resolve_k_points(raw, dataset_config, nk=bundle.nk)
    solver_config = _build_solver_config(raw, dataset_config, nb=bundle.nb)

    wall_start = time.perf_counter()
    if any(_as_mapping(record.get("strategy_rankings")) for _source_index, _stratum, record in target_records):
        metrics = _run_strategy_replay_payload(
            cfg=cfg,
            run_dir=run_dir,
            candidates_path=candidates_path,
            target_records=target_records,
            target_shape=target_shape,
            nb=int(bundle.nb),
            replay_top_m=int(replay_top_m),
            backend=backend,
            k_points=k_points,
            solver_config=solver_config,
            solver_fn=solver_fn,
            band_range_2_10=band_range_2_10,
            band_range_1_10=band_range_1_10,
            primary_range=primary_range,
            gap_threshold=gap_threshold,
            validation_boundary=validation_boundary,
            timeout_seconds=timeout_seconds,
            wall_start=wall_start,
            n_targets_cfg=int(n_targets_cfg),
            replay_cfg=replay_cfg,
            k_path_meta=k_path_meta,
        )
        try:
            plot_metric_report(metrics, run_dir / "figures" / "metrics.png", title="inverse strategy MPB replay metrics")
        except Exception:
            logger.warning("Could not write replay metric figure.")
        logger.info("Replay mpb finished: %s", run_dir)
        return run_dir

    target_rows: list[dict[str, Any]] = []
    candidate_jsonl_rows: list[dict[str, Any]] = []
    best_band_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    gap_true: list[bool] = []
    gap_pred: list[bool] = []
    success_count = 0
    selected_target_bands: list[np.ndarray] = []
    selected_surrogate_bands: list[np.ndarray | None] = []
    selected_replay_bands: list[np.ndarray] = []
    selected_surrogate_sources: list[str] = []

    for local_pos, (source_index, stratum, target_record) in enumerate(target_records):
        start = time.perf_counter()
        target_id = str(target_record.get("target_id", f"target_{source_index:06d}"))
        condition = _as_mapping(target_record.get("condition"))
        target_matrix = _matrix_from_condition(
            condition.get(cfg.condition_target, condition.get("target", [])),
            target_shape=target_shape,
            target_id=target_id,
        )
        candidates, selected_indices = _select_candidates(target_record, top_m=replay_top_m)
        if candidates.size == 0:
            raise ValueError(f"Target {target_id} has no candidates")

        replay_pred, solver_metadata = _run_candidate_replay(
            candidates,
            target_matrix,
            backend=backend,
            seed=_derive_seed(cfg.seed, target_id, local_pos),
            k_points=k_points,
            solver_config=solver_config,
            solver_fn=solver_fn,
        )
        success_count += 1
        surrogate_pred, surrogate_source = _extract_surrogate_predictions(
            target_record,
            selected_indices=selected_indices,
            target_shape=target_shape,
        )

        replay_2_10, best_idx = _best_error(replay_pred, target_matrix, band_range_2_10)
        replay_1_10, _ = _best_error(replay_pred, target_matrix, band_range_1_10)
        replay_primary, best_primary_idx = _best_error(replay_pred, target_matrix, primary_range)
        if best_primary_idx != best_idx:
            best_idx = best_primary_idx
        surrogate_2_10: float | None
        surrogate_1_10: float | None
        if surrogate_pred is not None:
            surrogate_2_10, _ = _best_error(surrogate_pred, target_matrix, band_range_2_10)
            surrogate_1_10, _ = _best_error(surrogate_pred, target_matrix, band_range_1_10)
        else:
            surrogate_2_10 = _scalar_surrogate_mre(target_record, "best_of_k_surrogate_mre_bands_2_10")
            surrogate_1_10 = _scalar_surrogate_mre(target_record, "best_of_k_surrogate_mre_bands_1_10")

        best_replay = replay_pred[best_idx]
        best_surrogate = surrogate_pred[best_idx] if surrogate_pred is not None and best_idx < surrogate_pred.shape[0] else None
        selected_target_bands.append(np.asarray(target_matrix, dtype=np.float32))
        selected_replay_bands.append(np.asarray(best_replay, dtype=np.float32))
        selected_surrogate_bands.append(
            np.asarray(best_surrogate, dtype=np.float32) if best_surrogate is not None else None
        )
        selected_surrogate_sources.append(str(surrogate_source))
        target_labels, target_ratios = _complete_gap_labels(target_matrix, threshold_midgap_ratio=gap_threshold)
        replay_labels, replay_ratios = _complete_gap_labels(best_replay, threshold_midgap_ratio=gap_threshold)
        for gap_idx in range(min(len(target_labels), len(replay_labels))):
            gap_true.append(bool(target_labels[gap_idx]))
            gap_pred.append(bool(replay_labels[gap_idx]))
            gap_rows.append(
                {
                    "target_id": target_id,
                    "target_index": int(target_record.get("target_index", source_index)),
                    "gap_pair_lower_band_one_based": int(gap_idx + 1),
                    "gap_pair_upper_band_one_based": int(gap_idx + 2),
                    "target_gap_label": bool(target_labels[gap_idx]),
                    "replay_gap_label": bool(replay_labels[gap_idx]),
                    "target_gap_midgap_ratio": float(target_ratios[gap_idx]),
                    "replay_gap_midgap_ratio": float(replay_ratios[gap_idx]),
                }
            )

        band_rows_for_target: list[dict[str, Any]] = []
        for band_idx in range(int(bundle.nb)):
            replay_band = best_replay[:, band_idx]
            target_band = target_matrix[:, band_idx]
            surrogate_band_mre: float | None = None
            surrogate_band_source = surrogate_source
            if best_surrogate is not None:
                surrogate_band_mre = float(np.mean(_relative_mre(best_surrogate[:, band_idx], target_band)) * 100.0)
                surrogate_band_source = "prediction"
            elif (band_idx + 1) >= band_range_2_10[0] and (band_idx + 1) <= band_range_2_10[1] and surrogate_2_10 is not None:
                surrogate_band_mre = float(surrogate_2_10 * 100.0)
                surrogate_band_source = "target_scalar_bands_2_10_broadcast"
            elif surrogate_1_10 is not None:
                surrogate_band_mre = float(surrogate_1_10 * 100.0)
                surrogate_band_source = "target_scalar_bands_1_10_broadcast"
            row = {
                "target_id": target_id,
                "target_index": int(target_record.get("target_index", source_index)),
                "stratum": stratum,
                "band_one_based": int(band_idx + 1),
                "selected_candidate_index": int(selected_indices[best_idx]) if best_idx < len(selected_indices) else int(best_idx),
                "surrogate_mre_pct": surrogate_band_mre,
                "surrogate_mre_source": surrogate_band_source,
                "mpb_replay_mre_pct": float(np.mean(_relative_mre(replay_band, target_band)) * 100.0),
                "surrogate_replay_gap_pct": None if surrogate_band_mre is None else float(np.mean(_relative_mre(replay_band, target_band)) * 100.0 - surrogate_band_mre),
                "mae": float(np.mean(_mae(replay_band, target_band))),
                "rmse": _rmse(replay_band, target_band),
                "sample_count": int(target_band.shape[0]),
            }
            band_rows_for_target.append(row)
            best_band_rows.append(row)

        elapsed = float(time.perf_counter() - start)
        target_row = {
            "stage": "replay",
            "target_id": target_id,
            "target_index": int(target_record.get("target_index", source_index)),
            "sample_id": str(target_record.get("sample_id", target_id)),
            "dataset_id": str(target_record.get("dataset_id", "")),
            "split": str(target_record.get("split", "unassigned")),
            "stratum": stratum,
            "candidate_count_selected": int(len(selected_indices)),
            "selected_indices": selected_indices,
            "best_candidate_selected_index": int(selected_indices[best_idx]) if best_idx < len(selected_indices) else int(best_idx),
            "best_of_k_replay_mre_bands_2_10_pct": float(replay_2_10 * 100.0),
            "best_of_k_replay_mre_bands_1_10_pct": float(replay_1_10 * 100.0),
            "best_of_k_replay_mre_primary_pct": float(replay_primary * 100.0),
            "best_of_k_surrogate_mre_bands_2_10_pct": None if surrogate_2_10 is None else float(surrogate_2_10 * 100.0),
            "best_of_k_surrogate_mre_bands_1_10_pct": None if surrogate_1_10 is None else float(surrogate_1_10 * 100.0),
            "surrogate_replay_gap_bands_2_10_pct": None if surrogate_2_10 is None else float(replay_2_10 * 100.0 - surrogate_2_10 * 100.0),
            "surrogate_replay_gap_bands_1_10_pct": None if surrogate_1_10 is None else float(replay_1_10 * 100.0 - surrogate_1_10 * 100.0),
            "mpb_replay_time_s": elapsed,
            "mpb_replay_time_per_candidate_s": elapsed / max(int(len(selected_indices)), 1),
            "surrogate_mre_source": surrogate_source,
        }
        target_rows.append(target_row)
        for candidate_pos, selected in enumerate(selected_indices):
            candidate_jsonl_rows.append(
                {
                    "target_id": target_id,
                    "target_index": int(target_record.get("target_index", source_index)),
                    "candidate_local_pos": int(candidate_pos),
                    "selected_candidate_index": int(selected),
                    "solver_metadata": solver_metadata[candidate_pos] if candidate_pos < len(solver_metadata) else {},
                }
            )

    target_count = len(target_rows)
    replay_success_rate = float(success_count / max(target_count, 1))
    gap_metrics = _classification_metrics(gap_true, gap_pred)

    def mean_key(key: str) -> float | None:
        values = [row.get(key) for row in target_rows]
        numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        return float(np.mean(numbers)) if numbers else None

    metrics = {
        "best_of_k_replay_mre_bands_2_10_pct": mean_key("best_of_k_replay_mre_bands_2_10_pct"),
        "best_of_k_replay_mre_bands_1_10_pct": mean_key("best_of_k_replay_mre_bands_1_10_pct"),
        "best_of_k_surrogate_mre_bands_2_10_pct": mean_key("best_of_k_surrogate_mre_bands_2_10_pct"),
        "best_of_k_surrogate_mre_bands_1_10_pct": mean_key("best_of_k_surrogate_mre_bands_1_10_pct"),
        "surrogate_replay_gap_bands_2_10_pct": mean_key("surrogate_replay_gap_bands_2_10_pct"),
        "surrogate_replay_gap_bands_1_10_pct": mean_key("surrogate_replay_gap_bands_1_10_pct"),
        "mpb_replay_success_rate": replay_success_rate,
        "mpb_replay_time_s": float(sum(float(row["mpb_replay_time_s"]) for row in target_rows)),
        "mpb_replay_time_per_candidate_s": float(sum(float(row["mpb_replay_time_s"]) for row in target_rows) / max(sum(int(row["candidate_count_selected"]) for row in target_rows), 1)),
        "wall_time_total_s": float(time.perf_counter() - wall_start),
        "targets": int(target_count),
        "replay_timeout_seconds": timeout_seconds,
        "replay_top_m": int(replay_top_m),
        "replay_backend": backend,
        "validation_boundary": validation_boundary,
        "selection_unit": selection_unit,
        "within_target_record_policy": within_target_record_policy,
        "unique_target_index_count": int(
            len({_target_index(record, source_index) for source_index, _stratum, record in target_records})
        ),
        "target_indices_path": target_indices_path or None,
        "write_target_indices_path": write_target_indices_path or None,
        **gap_metrics,
    }
    write_metrics(run_dir, metrics)

    per_band_rows = _aggregate_per_band(best_band_rows, nb=bundle.nb)
    _write_csv(run_dir / "history.csv", target_rows)
    _write_csv(run_dir / "per_target_replay_metrics.csv", target_rows)
    _write_csv(run_dir / "per_band_replay_metrics.csv", per_band_rows)
    _write_csv(run_dir / "per_target_band_replay_metrics.csv", best_band_rows)
    _write_csv(run_dir / "gap_metrics.csv", gap_rows)
    _write_jsonl(run_dir / "replay_candidates.jsonl", candidate_jsonl_rows)
    band_artifacts = _save_selected_band_artifacts(
        run_dir,
        target_bands=selected_target_bands,
        surrogate_pred_bands=selected_surrogate_bands,
        mpb_replay_bands=selected_replay_bands,
        surrogate_sources=selected_surrogate_sources,
    )

    write_results(
        run_dir,
        {
            "action": "replay mpb",
            "stage": "inverse_candidate_replay",
            "run_dir": str(run_dir),
            "experiment_id": cfg.experiment_id,
            "candidates_path": str(candidates_path),
            "dataset_path": str(cfg.dataset_path),
            "validation_boundary": validation_boundary,
            "replay_backend": backend,
            "band_ranges": {
                "bands_1_10": list(band_range_1_10),
                "bands_2_10": list(band_range_2_10),
                "primary": list(primary_range),
                "band_indexing": "one_based_inclusive",
            },
            "k_path": k_path_meta,
            "target_shape": list(target_shape),
            "targets": target_rows,
            "per_band_metrics_path": "per_band_replay_metrics.csv",
            "per_target_metrics_path": "per_target_replay_metrics.csv",
            "gap_metrics_path": "gap_metrics.csv",
            "replay_candidates_path": "replay_candidates.jsonl",
            "band_array_artifacts": band_artifacts,
            "n_targets_requested": int(n_targets_cfg),
            "n_targets_processed": int(target_count),
            "selection_unit": selection_unit,
            "within_target_record_policy": within_target_record_policy,
            "target_indices_path": target_indices_path or None,
            "write_target_indices_path": write_target_indices_path or None,
            "replay_enabled": bool(replay_cfg.get("enabled", True)),
        },
    )

    try:
        plot_metric_report(metrics, run_dir / "figures" / "metrics.png", title="inverse MPB replay metrics")
    except Exception:
        logger.warning("Could not write replay metric figure.")
    logger.info("Replay mpb finished: %s", run_dir)
    return run_dir


__all__ = [
    "run_replay_mpb",
    "_classification_metrics",
    "_complete_gap_labels",
    "_select_target_records",
]
