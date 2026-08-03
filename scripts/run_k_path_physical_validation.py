"""Run five-sample real-MPB k-path physical validation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.data.forward_dataset import parse_metadata_jsonl
from phc2d.data.k_path import KPathMetadata, hash_k_path_points, hash_k_path_spec, load_k_path_spec, save_k_path_spec
from phc2d.runs import ensure_figures_dir, make_run_dir, setup_logging, write_metrics, write_resolved_config, write_results, write_run_meta
from phc2d.solvers import MPBParallelConfig, MPBWorkerJob, MpbBandSolverConfig, run_mpb_worker_jobs, solve_bands_for_rho_mpb


ROOT = Path(__file__).resolve().parents[1]
CRITICAL_SOLVER_FIELDS = (
    "lattice",
    "reciprocal_basis",
    "epsilon_air",
    "epsilon_material",
    "geometry_lifting_convention",
    "grid_shape",
    "n_wedge",
    "resolution",
    "polarization",
    "num_bands",
    "band_guard_count",
    "band_order_policy",
    "eigensolver_tolerance",
    "k_path",
    "endpoint_policy",
    "frequency_normalization",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _jsonable(row.get(name, "")) for name in fieldnames})


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    return _sha256_bytes(arr.tobytes())


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return payload


def _load_embedded_config(dataset: Mapping[str, np.ndarray]) -> dict[str, Any]:
    raw = dataset.get("config_yaml")
    if raw is None:
        return {}
    payload = yaml.safe_load(str(raw)) or {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _resolve_existing_path(value: str | Path, candidates: Sequence[Path]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in candidates:
        candidate = base / path
        if candidate.exists():
            return candidate
    return candidates[0] / path


def _resolve_optional_existing_path(value: str | Path | None, candidates: Sequence[Path]) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_existing_path(str(value), candidates)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"YAML file must contain a mapping: {path}")
    return dict(payload)


def _write_yaml_mapping(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_jsonable(payload), sort_keys=False, allow_unicode=True), encoding="utf-8")


def _solver_parallel_config(
    solver_cfg: Mapping[str, Any],
    *,
    workers: int | str | None = None,
    min_workers: int | None = None,
    max_workers: int | None = None,
    worker_backend: str | None = None,
    force_serial_default: bool = False,
) -> MPBParallelConfig:
    payload = dict(solver_cfg)
    if force_serial_default and "workers" not in payload and workers is None:
        payload["workers"] = 1
        payload["min_workers"] = 1
        payload["max_workers"] = 1
    if workers is not None:
        payload["workers"] = workers
    if min_workers is not None:
        payload["min_workers"] = min_workers
    if max_workers is not None:
        payload["max_workers"] = max_workers
    if worker_backend is not None:
        payload["worker_backend"] = worker_backend
    return MPBParallelConfig.from_mapping(payload)


def _manifest_path(dataset_cfg: Mapping[str, Any], dataset_path: Path) -> Path | None:
    value = dataset_cfg.get("manifest")
    return _resolve_optional_existing_path(str(value), [dataset_path.parent / "metadata", dataset_path.parent, ROOT]) if value else None


def _manifest_solver_provenance_path(manifest_path: Path | None, dataset_path: Path) -> Path | None:
    if manifest_path is None or not manifest_path.exists():
        return None
    manifest = _read_yaml_mapping(manifest_path)
    value = manifest.get("solver_provenance")
    if not value:
        return None
    return _resolve_existing_path(str(value), [ROOT, dataset_path.parent, dataset_path.parent / "metadata", manifest_path.parent])


def _promote_verified_sidecar(
    *,
    cfg: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
    dataset_path: Path,
    sidecar_path: Path,
    manifest_path: Path | None,
    run_dir: Path,
    decision: str,
    accepted_tolerance: float | None,
) -> tuple[str, str]:
    output_cfg = dict(cfg.get("output") or {})
    if not bool(output_cfg.get("promote_verified_sidecar", False)):
        return "not_requested", ""
    if decision not in {"physically_verified", "physically_verified_with_documented_numerical_tolerance"}:
        return "not_promoted", ""

    target_value = output_cfg.get("promoted_k_path_sidecar")
    promoted_path = _resolve_path(str(target_value)) if target_value else sidecar_path
    sidecar = load_k_path_spec(sidecar_path)
    notes = tuple(sidecar.metadata.notes) + (
        f"Promoted by physical validation run {run_dir.name}.",
        f"decision={decision}; accepted_tolerance={accepted_tolerance}",
    )
    promoted = replace(
        sidecar,
        metadata=KPathMetadata(
            source_type=sidecar.metadata.source_type,
            source_path=sidecar.metadata.source_path,
            source_commit=sidecar.metadata.source_commit,
            source_run_id=sidecar.metadata.source_run_id,
            source_config_hash=sidecar.metadata.source_config_hash,
            verification_status=decision,
            verification_run_id=run_dir.name,
            notes=notes,
        ),
    )
    save_k_path_spec(promoted, promoted_path)

    if manifest_path is not None:
        manifest = _read_yaml_mapping(manifest_path)
        manifest["k_path_sidecar"] = promoted_path.name if promoted_path.parent == manifest_path.parent else _display_path(promoted_path)
        manifest["physical_verification_status"] = decision
        manifest["physical_verification_run_id"] = run_dir.name
        manifest["physical_verification_run"] = _display_path(run_dir)
        manifest["accepted_tolerance"] = accepted_tolerance
        manifest["formal_full_2a_allowed"] = True
        manifest["dataset_path"] = manifest.get("dataset_path") or _display_path(dataset_path)
        _write_yaml_mapping(manifest_path, manifest)

    return "promoted", _display_path(promoted_path)


def _field(
    value: Any,
    *,
    source_path: str,
    source_commit: str,
    confidence: str,
    verified: bool,
    status: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    return {
        "value": value,
        "source_path": source_path,
        "source_commit_or_run": source_commit,
        "confidence": confidence,
        "whether_verified": bool(verified),
        "status": status or ("available" if verified else "provenance_uncertain"),
        "note": note,
    }


def _scalar_or_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("value")
    return value


def _load_npz_metadata(data: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    if "metadata_jsonl" not in data:
        return []
    return parse_metadata_jsonl(data["metadata_jsonl"])


def _sample_split_indices(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "train": np.asarray(data.get("idx_train", np.array([], dtype=np.int64)), dtype=np.int64),
        "val": np.asarray(data.get("idx_val", np.array([], dtype=np.int64)), dtype=np.int64),
        "test": np.asarray(data.get("idx_test", np.array([], dtype=np.int64)), dtype=np.int64),
    }


def _dataset_identity_mismatch_reason(
    dataset_cfg: Mapping[str, Any],
    dataset_path: Path,
    data: Mapping[str, np.ndarray],
) -> str | None:
    """Return a frozen-protocol identity mismatch before MPB is invoked."""
    expected_hash = str(dataset_cfg.get("expected_dataset_sha256") or "").strip()
    if expected_hash:
        actual_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            return f"dataset_sha256_mismatch expected={expected_hash} actual={actual_hash}"
    expected_splits = dataset_cfg.get("expected_split_sizes")
    if isinstance(expected_splits, Mapping):
        actual_splits = {name: int(indices.shape[0]) for name, indices in _sample_split_indices(data).items()}
        for name in ("train", "val", "test"):
            if name in expected_splits and int(expected_splits[name]) != actual_splits[name]:
                return f"split_size_mismatch split={name} expected={expected_splits[name]} actual={actual_splits[name]}"
    return None


def _select_samples(data: Mapping[str, np.ndarray], selection_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    split_indices = _sample_split_indices(data)
    if str(selection_cfg.get("strategy", "")).lower() == "all_available_smoke_samples":
        split_counts = {split: int(indices.shape[0]) for split, indices in split_indices.items()}
    else:
        split_counts = {
            "train": int(selection_cfg.get("train_count", 2)),
            "val": int(selection_cfg.get("val_count", 1)),
            "test": int(selection_cfg.get("test_count", 2)),
        }
    metadata_rows = _load_npz_metadata(data)
    rows: list[dict[str, Any]] = []
    rho = np.asarray(data["rho"])
    wh = np.asarray(data["Wh"])
    for split, count in split_counts.items():
        indices = split_indices[split]
        if indices.shape[0] < count:
            raise ValueError(f"Split {split} has only {indices.shape[0]} samples, cannot select {count}")
        for pos, dataset_index in enumerate(indices[:count]):
            idx = int(dataset_index)
            rho_row = np.asarray(rho[idx])
            wh_row = np.asarray(wh[idx])
            metadata = metadata_rows[idx] if idx < len(metadata_rows) and isinstance(metadata_rows[idx], Mapping) else {}
            fill_fraction = float(np.mean(rho_row))
            rows.append(
                {
                    "sample_id": str(metadata.get("sample_id") or idx),
                    "dataset_index": idx,
                    "split": str(metadata.get("split") or split),
                    "split_position": int(pos),
                    "rho_hash": str(metadata.get("rho_hash") or _hash_array(rho_row)),
                    "geometry_hash": str(metadata.get("geometry_hash") or _hash_array((rho_row > 0.5).astype(np.uint8))),
                    "existing_Wh_hash": str(metadata.get("Wh_hash") or _hash_array(wh_row)),
                    "fill_fraction": fill_fraction,
                    "selection_reason": str(selection_cfg.get("rule") or "first_n_from_embedded_fixed_split; richer morphology/gap strata unavailable"),
                }
            )
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON provenance must contain a mapping: {path}")
    return dict(payload)


def _provenance_field_value(provenance: Mapping[str, Any], name: str) -> Any:
    fields = provenance.get("fields")
    if isinstance(fields, Mapping) and name in fields:
        return _scalar_or_none(fields[name])
    return provenance.get(name)


def _field_from_provenance(
    provenance: Mapping[str, Any],
    name: str,
    *,
    source_path: str,
    source_commit: str,
    note: str = "",
) -> dict[str, Any]:
    fields = provenance.get("fields")
    raw = fields.get(name) if isinstance(fields, Mapping) else None
    value = _scalar_or_none(raw) if raw is not None else provenance.get(name)
    status = str(raw.get("status")) if isinstance(raw, Mapping) and raw.get("status") is not None else ("available" if value is not None else "missing")
    return _field(
        value,
        source_path=str(raw.get("source", source_path)) if isinstance(raw, Mapping) else source_path,
        source_commit=source_commit,
        confidence="high" if status == "available" else "missing",
        verified=status == "available" and value is not None,
        status=status,
        note=note,
    )


def _build_v2_solver_provenance(
    provenance_path: Path,
    sidecar_path: Path,
) -> dict[str, Any]:
    source = _read_json(provenance_path)
    sidecar = load_k_path_spec(sidecar_path)
    source_commit = str(_provenance_field_value(source, "source_code_commit") or sidecar.metadata.source_commit or "")
    source_path = _display_path(provenance_path)
    sidecar_display = _display_path(sidecar_path)
    fields = {
        "lattice": _field_from_provenance(source, "lattice", source_path=source_path, source_commit=source_commit),
        "reciprocal_basis": _field_from_provenance(source, "reciprocal_basis", source_path=source_path, source_commit=source_commit),
        "epsilon_air": _field_from_provenance(source, "epsilon_air", source_path=source_path, source_commit=source_commit),
        "epsilon_material": _field_from_provenance(source, "epsilon_material", source_path=source_path, source_commit=source_commit),
        "geometry_lifting_convention": _field_from_provenance(source, "geometry_lifting_convention", source_path=source_path, source_commit=source_commit),
        "grid_shape": _field_from_provenance(source, "grid_shape", source_path=source_path, source_commit=source_commit),
        "n_wedge": _field_from_provenance(source, "n_wedge", source_path=source_path, source_commit=source_commit),
        "resolution": _field_from_provenance(source, "resolution", source_path=source_path, source_commit=source_commit),
        "polarization": _field_from_provenance(source, "polarization", source_path=source_path, source_commit=source_commit),
        "num_bands": _field_from_provenance(source, "num_bands", source_path=source_path, source_commit=source_commit),
        "band_guard_count": _field(
            int(_provenance_field_value(source, "band_guard_count") or 0),
            source_path=source_path,
            source_commit=source_commit,
            confidence="high",
            verified=True,
            note="Defaults to 0 for pre-guard V2 provenance; guarded V2 runs record a positive value.",
        ),
        "band_order_policy": _field(
            str(_provenance_field_value(source, "band_order_policy") or "mpb_output"),
            source_path=source_path,
            source_commit=source_commit,
            confidence="high",
            verified=bool(_provenance_field_value(source, "band_order_policy")),
            note="V2 canonical datasets must record frequency_ascending so generation and validation compare the same retained-band policy.",
        ),
        "eigensolver_tolerance": _field_from_provenance(
            source,
            "eigensolver_tolerance",
            source_path=source_path,
            source_commit=source_commit,
            note="Explicit V2 tolerance is required before replaying stored Wh.",
        ),
        "k_path": _field(
            {
                "path": sidecar_display,
                "k_path_id": sidecar.k_path_id,
                "Nk": sidecar.Nk,
                "k_points_hash": sidecar.k_points_hash or hash_k_path_points(sidecar.k_points),
                "k_path_hash": hash_k_path_spec(sidecar),
            },
            source_path=sidecar_display,
            source_commit=source_commit,
            confidence="high",
            verified=True,
        ),
        "endpoint_policy": _field(
            [segment.endpoint_policy for segment in sidecar.segments],
            source_path=sidecar_display,
            source_commit=source_commit,
            confidence="high",
            verified=bool(sidecar.segments),
        ),
        "frequency_normalization": _field_from_provenance(source, "frequency_normalization", source_path=source_path, source_commit=source_commit),
    }
    for name in CRITICAL_SOLVER_FIELDS:
        if not fields[name]["whether_verified"]:
            fields[name]["status"] = "missing"
    missing = [name for name in CRITICAL_SOLVER_FIELDS if fields[name]["status"] == "missing"]
    return {
        "solver_backend": str(source.get("solver_backend", "real_mpb")),
        "source_schema_version": source.get("schema_version"),
        "source_status": source.get("status"),
        "use_proxy": bool(source.get("use_proxy", False)),
        "critical_fields": list(CRITICAL_SOLVER_FIELDS),
        "fields": fields,
        "missing_critical_fields": missing,
        "solver_provenance_complete": not missing and bool(source.get("solver_provenance_complete", True)),
        "source_provenance_path": source_path,
    }


def _build_solver_provenance(
    cfg: Mapping[str, Any],
    embedded_cfg: Mapping[str, Any],
    sidecar_path: Path,
) -> dict[str, Any]:
    solver_cfg = dict(cfg.get("solver") or {})
    source_commit = "6c1b95e029441d1d8f47265f594cd87077e3663d"
    source_run = "runs/20260610/dataset_gen_paper_like_50k_202606102000"
    fem_cfg = dict(embedded_cfg.get("fem") or {})
    mpb_cfg = dict(embedded_cfg.get("mpb") or {})
    geom_cfg = dict(embedded_cfg.get("geometry") or {})
    k_cfg = dict(embedded_cfg.get("k_path") or {})
    sidecar = load_k_path_spec(sidecar_path)
    sidecar_display = _display_path(sidecar_path)

    tolerance_value = _scalar_or_none(solver_cfg.get("eigensolver_tolerance"))
    fields = {
        "lattice": _field(
            "square",
            source_path="embedded config lattice_constant + historical mpb_solver.py",
            source_commit=source_commit,
            confidence="high",
            verified=True,
        ),
        "reciprocal_basis": _field(
            "square reciprocal basis; MPB reduced coordinates",
            source_path=sidecar_display,
            source_commit=source_commit,
            confidence="high_candidate_unverified",
            verified=True,
        ),
        "epsilon_air": _field(
            embedded_cfg.get("epsilon_air"),
            source_path="data/paper_like_50k_mpb.npz config_yaml",
            source_commit=source_run,
            confidence="high",
            verified=embedded_cfg.get("epsilon_air") is not None,
        ),
        "epsilon_material": _field(
            embedded_cfg.get("epsilon_alum"),
            source_path="data/paper_like_50k_mpb.npz config_yaml",
            source_commit=source_run,
            confidence="high",
            verified=embedded_cfg.get("epsilon_alum") is not None,
        ),
        "geometry_lifting_convention": _field(
            {
                "grid_n": geom_cfg.get("grid_n"),
                "n_wedge": geom_cfg.get("n_wedge"),
                "builder": "p4m wedge -> binary grid -> occupied MPB Blocks",
            },
            source_path="embedded config + git:src/pod_deeponet/data_gen/lift.py + git:src/pod_deeponet/data_gen/mpb_solver.py",
            source_commit=source_commit,
            confidence="high_candidate_unverified",
            verified=geom_cfg.get("grid_n") is not None and geom_cfg.get("n_wedge") is not None,
        ),
        "grid_shape": _field(
            [geom_cfg.get("grid_n"), geom_cfg.get("grid_n")] if geom_cfg.get("grid_n") is not None else None,
            source_path="data/paper_like_50k_mpb.npz config_yaml geometry.grid_n",
            source_commit=source_run,
            confidence="high",
            verified=geom_cfg.get("grid_n") is not None,
        ),
        "n_wedge": _field(
            geom_cfg.get("n_wedge"),
            source_path="data/paper_like_50k_mpb.npz config_yaml geometry.n_wedge",
            source_commit=source_run,
            confidence="high",
            verified=geom_cfg.get("n_wedge") is not None,
        ),
        "resolution": _field(
            mpb_cfg.get("resolution"),
            source_path="data/paper_like_50k_mpb.npz config_yaml mpb.resolution",
            source_commit=source_run,
            confidence="high",
            verified=mpb_cfg.get("resolution") is not None,
        ),
        "polarization": _field(
            mpb_cfg.get("polarization", fem_cfg.get("polarization")),
            source_path="data/paper_like_50k_mpb.npz config_yaml mpb/fem polarization",
            source_commit=source_run,
            confidence="high",
            verified=bool(mpb_cfg.get("polarization") or fem_cfg.get("polarization")),
        ),
        "num_bands": _field(
            fem_cfg.get("n_bands"),
            source_path="data/paper_like_50k_mpb.npz config_yaml fem.n_bands",
            source_commit=source_run,
            confidence="high",
            verified=fem_cfg.get("n_bands") is not None,
        ),
        "band_guard_count": _field(
            int(solver_cfg.get("band_guard_count", 0)),
            source_path="validation config solver.band_guard_count default",
            source_commit=source_commit,
            confidence="high",
            verified=True,
            note="Optional guard bands for MPB top retained band stability.",
        ),
        "band_order_policy": _field(
            str(solver_cfg.get("band_order_policy", "mpb_output")),
            source_path="validation config solver.band_order_policy default",
            source_commit=source_commit,
            confidence="high_candidate_unverified",
            verified=bool(solver_cfg.get("band_order_policy")),
            note="Legacy candidate datasets without an explicit policy are not promoted as canonical V2.",
        ),
        "eigensolver_tolerance": _field(
            tolerance_value,
            source_path="not recorded in embedded config, run.meta.json, or source commit search",
            source_commit=source_commit,
            confidence="missing",
            verified=tolerance_value is not None,
            status="missing" if tolerance_value is None else "available",
            note="Formal validation cannot invoke MPB with an unrecorded library default.",
        ),
        "k_path": _field(
            {
                "path": sidecar_display,
                "k_path_id": sidecar.k_path_id,
                "Nk": sidecar.Nk,
                "k_points_hash": sidecar.k_points_hash or hash_k_path_points(sidecar.k_points),
                "k_path_hash": hash_k_path_spec(sidecar),
            },
            source_path=sidecar_display,
            source_commit=source_commit,
            confidence="high_candidate_unverified",
            verified=True,
        ),
        "endpoint_policy": _field(
            [segment.endpoint_policy for segment in sidecar.segments],
            source_path=sidecar_display,
            source_commit=source_commit,
            confidence="high_candidate_unverified",
            verified=bool(sidecar.segments),
        ),
        "frequency_normalization": _field(
            "MPB dimensionless frequency a/lambda, a=1; compared directly to stored Wh",
            source_path="git:src/pod_deeponet/data_gen/mpb_solver.py and solver.py comments",
            source_commit=source_commit,
            confidence="high_candidate_unverified",
            verified=True,
        ),
    }

    for name in CRITICAL_SOLVER_FIELDS:
        if not fields[name]["whether_verified"]:
            fields[name]["status"] = "missing"
    missing = [name for name in CRITICAL_SOLVER_FIELDS if fields[name]["status"] == "missing"]
    return {
        "solver_backend": "real_mpb_candidate",
        "use_proxy": bool(solver_cfg.get("use_proxy", False)),
        "critical_fields": list(CRITICAL_SOLVER_FIELDS),
        "fields": fields,
        "missing_critical_fields": missing,
        "solver_provenance_complete": not missing,
    }


def _metric_block(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = pred - target
    abs_delta = np.abs(delta)
    denom = np.maximum(np.abs(target), 1.0e-12)
    bands_2_10 = slice(1, min(10, target.shape[1]))
    return {
        "direct_alignment_rmse": float(np.sqrt(np.mean(delta**2))),
        "direct_alignment_mae": float(np.mean(abs_delta)),
        "max_abs_error": float(np.max(abs_delta)),
        "avg_MRE_bands_2_10_pct": float(100.0 * np.mean(abs_delta[:, bands_2_10] / denom[:, bands_2_10])),
        "per_k_rmse": np.sqrt(np.mean(delta**2, axis=1)).tolist(),
        "per_band_rmse": np.sqrt(np.mean(delta**2, axis=0)).tolist(),
    }


def _alignment_diagnostics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    def rmse(arr: np.ndarray, ref: np.ndarray = target) -> float:
        if arr.shape != ref.shape:
            return float("nan")
        return float(np.sqrt(np.mean((arr - ref) ** 2)))

    diagnostics = {
        "direct_alignment": {"rmse": rmse(pred), "used_for_pass_fail": True},
        "reversed_full_k_order": {"rmse": rmse(pred[::-1]), "used_for_pass_fail": False},
        "reversed_segment_order": {"status": "not_auto_applied", "used_for_pass_fail": False},
        "shared_endpoint_duplicate_removal_difference": {"status": "not_auto_applied", "used_for_pass_fail": False},
        "one_index_k_offset_minus": {"rmse": rmse(pred[1:], target[:-1]) if pred.shape[0] > 1 else float("nan"), "used_for_pass_fail": False},
        "one_index_k_offset_plus": {"rmse": rmse(pred[:-1], target[1:]) if pred.shape[0] > 1 else float("nan"), "used_for_pass_fail": False},
        "band_index_offset_minus": {"rmse": rmse(pred[:, 1:], target[:, :-1]) if pred.shape[1] > 1 else float("nan"), "used_for_pass_fail": False},
        "band_index_offset_plus": {"rmse": rmse(pred[:, :-1], target[:, 1:]) if pred.shape[1] > 1 else float("nan"), "used_for_pass_fail": False},
        "te_tm_mismatch": {"status": "requires_explicit_second_polarization_run", "used_for_pass_fail": False},
        "frequency_normalization_mismatch": {"status": "diagnostic_only_no_auto_scaling", "used_for_pass_fail": False},
        "geometry_inversion_material_swap": {"status": "requires_explicit_alternative_geometry_run", "used_for_pass_fail": False},
        "rho_lifting_mismatch": {"status": "requires_authoritative_lifting_audit", "used_for_pass_fail": False},
    }
    return diagnostics


def _write_blocked_outputs(
    run_dir: Path,
    *,
    config_path: Path,
    resolved_yaml: str,
    selected_samples: list[dict[str, Any]],
    solver_provenance: dict[str, Any],
    status: str,
    reason: str,
) -> None:
    ensure_figures_dir(run_dir)
    (run_dir / "recomputed_bands").mkdir(exist_ok=True)
    (run_dir / "stored_bands").mkdir(exist_ok=True)
    write_resolved_config(run_dir, resolved_yaml)
    write_run_meta(
        run_dir,
        {
            "script": "scripts/run_k_path_physical_validation.py",
            "config_path": str(config_path),
            "validation_boundary": "wsl_meep_env_mpb_physical_validation",
            "status": status,
        },
    )
    _write_json(run_dir / "solver_provenance.json", solver_provenance)
    _write_json(run_dir / "selected_samples.json", {"samples": selected_samples})
    _write_csv(
        run_dir / "selected_samples.csv",
        selected_samples,
        ["sample_id", "dataset_index", "split", "split_position", "rho_hash", "geometry_hash", "existing_Wh_hash", "fill_fraction", "selection_reason"],
    )
    empty_metrics = {
        "physical_verification_decision": status,
        "real_mpb_execution_started": False,
        "real_mpb_execution_completed": False,
        "direct_alignment_verified": False,
        "tolerance_justified": False,
        "missing_critical_fields": solver_provenance.get("missing_critical_fields", []),
        "wall_time_total_s": 0.0,
    }
    write_metrics(run_dir, empty_metrics)
    write_results(
        run_dir,
        {
            "action": "k_path_physical_validation_5sample",
            "status": status,
            "reason": reason,
            "formal_full_2a_allowed": False,
            "sidecar_manifest_status": "unchanged_candidate_unverified",
        },
    )
    _write_csv(run_dir / "per_sample_metrics.csv", [], ["sample_id", "split", "status", "direct_alignment_rmse", "max_abs_error"])
    _write_csv(run_dir / "per_k_metrics.csv", [], ["sample_id", "k_index", "rmse", "mae"])
    _write_csv(run_dir / "per_band_metrics.csv", [], ["sample_id", "band_index", "rmse", "mae"])
    _write_csv(run_dir / "failure_cases.csv", [{"status": status, "reason": reason}], ["status", "reason"])
    _write_json(run_dir / "alignment_diagnostics.json", {"status": "not_run", "reason": reason})
    (run_dir / "history.csv").write_text("stage,status,reason\nprovenance_check,blocked," + reason.replace(",", ";") + "\n", encoding="utf-8")
    (run_dir / "run.log").write_text(f"Physical validation blocked: {reason}\n", encoding="utf-8")


def run_validation(
    config_path: str | Path,
    *,
    run_dir_override: str | Path | None = None,
    workers: int | str | None = None,
    min_workers: int | None = None,
    max_workers: int | None = None,
    worker_backend: str | None = None,
    solver_fn=solve_bands_for_rho_mpb,
) -> Path:
    return _run_validation(
        config_path,
        run_dir_override=run_dir_override,
        workers=workers,
        min_workers=min_workers,
        max_workers=max_workers,
        worker_backend=worker_backend,
        solver_fn=solver_fn,
    )


def _run_validation(
    config_path: str | Path,
    *,
    run_dir_override: str | Path | None = None,
    workers: int | str | None = None,
    min_workers: int | None = None,
    max_workers: int | None = None,
    worker_backend: str | None = None,
    solver_fn=solve_bands_for_rho_mpb,
) -> Path:
    wall_start = time.perf_counter()
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    dataset_cfg = dict(cfg.get("dataset") or {})
    solver_cfg = dict(cfg.get("solver") or {})
    comparison_cfg = dict(cfg.get("comparison") or {})
    experiment_id = str(cfg.get("experiment_id", "k_path_physical_validation_5sample"))
    runs_root = Path(str((cfg.get("output") or {}).get("runs_dir", "runs")))
    run_dir = make_run_dir(experiment_id, runs_dir=runs_root, run_dir=run_dir_override)
    logger = setup_logging(
        log_file=run_dir / "run.log",
        run_meta={
            "script": "scripts/run_k_path_physical_validation.py",
            "config_path": str(config_path),
            "validation_boundary": "wsl_meep_env_mpb_physical_validation",
        },
    )
    resolved_yaml = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)

    if str(solver_cfg.get("backend", "")).lower() != "mpb":
        raise ValueError("solver.backend must be mpb for physical validation")
    if bool(solver_cfg.get("use_proxy", True)):
        raise ValueError("solver.use_proxy must be false; proxy replay is forbidden for this validation")

    dataset_path = _resolve_path(str(dataset_cfg["path"]))
    sidecar_path = _resolve_existing_path(
        str(dataset_cfg["k_path_sidecar"]),
        [dataset_path.parent / "metadata", dataset_path.parent, ROOT],
    )
    manifest_path = _manifest_path(dataset_cfg, dataset_path)
    provenance_value = dataset_cfg.get("solver_provenance") or solver_cfg.get("provenance_path")
    provenance_path = _resolve_optional_existing_path(
        str(provenance_value) if provenance_value else None,
        [ROOT, dataset_path.parent, dataset_path.parent / "metadata"],
    )
    if provenance_path is None:
        provenance_path = _manifest_solver_provenance_path(manifest_path, dataset_path)
    with np.load(dataset_path, allow_pickle=False) as data:
        embedded_cfg = _load_embedded_config(data)
        selected_samples = _select_samples(data, dict(cfg.get("sample_selection") or {}))
        solver_provenance = (
            _build_v2_solver_provenance(provenance_path, sidecar_path)
            if provenance_path is not None
            else _build_solver_provenance(cfg, embedded_cfg, sidecar_path)
        )
        identity_reason = _dataset_identity_mismatch_reason(dataset_cfg, dataset_path, data)
        if identity_reason is not None:
            _write_blocked_outputs(
                run_dir,
                config_path=config_path,
                resolved_yaml=resolved_yaml,
                selected_samples=selected_samples,
                solver_provenance=solver_provenance,
                status="dataset_identity_mismatch",
                reason=identity_reason,
            )
            logger.error(identity_reason)
            return run_dir
        if not solver_provenance["solver_provenance_complete"]:
            reason = "missing critical solver provenance: " + ", ".join(solver_provenance["missing_critical_fields"])
            _write_blocked_outputs(
                run_dir,
                config_path=config_path,
                resolved_yaml=resolved_yaml,
                selected_samples=selected_samples,
                solver_provenance=solver_provenance,
                status="insufficient_provenance",
                reason=reason,
            )
            logger.error(reason)
            return run_dir

        sidecar = load_k_path_spec(sidecar_path)
        grid_shape_value = solver_provenance["fields"]["grid_shape"]["value"]
        if not isinstance(grid_shape_value, Sequence) or len(grid_shape_value) != 2:
            raise ValueError(f"grid_shape must be a 2-item sequence, got {grid_shape_value!r}")
        solver = MpbBandSolverConfig(
            num_bands=int(solver_provenance["fields"]["num_bands"]["value"]),
            resolution=int(solver_provenance["fields"]["resolution"]["value"]),
            polarization=str(solver_provenance["fields"]["polarization"]["value"]),
            epsilon_air=float(solver_provenance["fields"]["epsilon_air"]["value"]),
            epsilon_material=float(solver_provenance["fields"]["epsilon_material"]["value"]),
            lattice_constant=float(embedded_cfg.get("lattice_constant", 1.0)),
            eigensolver_tolerance=float(solver_provenance["fields"]["eigensolver_tolerance"]["value"]),
            grid_shape=(int(grid_shape_value[0]), int(grid_shape_value[1])),
            band_guard_count=int(solver_provenance["fields"]["band_guard_count"]["value"]),
            band_order_policy=str(solver_provenance["fields"]["band_order_policy"]["value"]),
        )
        parallel_cfg = _solver_parallel_config(
            solver_cfg,
            workers=workers,
            min_workers=min_workers,
            max_workers=max_workers,
            worker_backend=worker_backend,
            force_serial_default=solver_fn is not solve_bands_for_rho_mpb,
        )

        ensure_figures_dir(run_dir)
        recomputed_dir = run_dir / "recomputed_bands"
        stored_dir = run_dir / "stored_bands"
        recomputed_dir.mkdir(exist_ok=True)
        stored_dir.mkdir(exist_ok=True)
        write_resolved_config(run_dir, resolved_yaml)
        _write_json(run_dir / "solver_provenance.json", solver_provenance)
        _write_json(run_dir / "selected_samples.json", {"samples": selected_samples})
        _write_csv(
            run_dir / "selected_samples.csv",
            selected_samples,
            ["sample_id", "dataset_index", "split", "split_position", "rho_hash", "geometry_hash", "existing_Wh_hash", "fill_fraction", "selection_reason"],
        )

        per_sample_rows: list[dict[str, Any]] = []
        per_k_rows: list[dict[str, Any]] = []
        per_band_rows: list[dict[str, Any]] = []
        diagnostics_payload: dict[str, Any] = {}
        sample_metrics: list[dict[str, Any]] = []
        repeat_variation: dict[str, Any] = {"status": "not_run"}
        accepted_tolerance = _scalar_or_none(comparison_cfg.get("accepted_tolerance"))
        failures: list[dict[str, Any]] = []
        jobs: list[MPBWorkerJob] = []
        stored_by_index: dict[int, np.ndarray] = {}
        for sample in selected_samples:
            idx = int(sample["dataset_index"])
            rho = np.asarray(data["rho"][idx], dtype=np.float64)
            stored = np.asarray(data["Wh"][idx], dtype=np.float64)
            stored_by_index[idx] = stored
            np.save(stored_dir / f"sample_{idx}_stored.npy", stored)
            jobs.append(
                MPBWorkerJob(
                    sample_id=str(sample["sample_id"]),
                    sample_index=idx,
                    split=str(sample["split"]),
                    rho=rho,
                    k_points=sidecar.k_points,
                    solver_config=solver,
                    rho_hash=str(sample.get("rho_hash", "")),
                    geometry_hash=str(sample.get("geometry_hash", "")),
                    stored=stored,
                )
            )

        worker_results, worker_summary = run_mpb_worker_jobs(
            jobs,
            config=parallel_cfg,
            shard_dir=run_dir / "worker_shards",
            solver_fn=solver_fn,
        )
        _write_json(run_dir / "worker_summary.json", worker_summary)
        samples_by_index = {int(sample["dataset_index"]): sample for sample in selected_samples}

        for result in worker_results:
            sample = samples_by_index.get(int(result.sample_index), {})
            if result.status != "computed" or result.bands is None:
                failures.append({"sample_id": result.sample_id, "split": result.split, "status": result.status, "reason": result.error})
                continue
            recomputed = np.asarray(result.bands, dtype=np.float64)
            stored = stored_by_index[int(result.sample_index)]
            sample["mpb_solve_metadata"] = result.solve_metadata
            solve_s = float(result.solve_time_s)
            idx = int(result.sample_index)
            np.save(recomputed_dir / f"sample_{idx}_recomputed.npy", recomputed)
            metrics = _metric_block(recomputed, stored)
            metrics.update({"sample_id": sample["sample_id"], "split": sample["split"], "mpb_replay_time_s": solve_s})
            sample_metrics.append(metrics)
            diagnostics_payload[sample["sample_id"]] = _alignment_diagnostics(recomputed, stored)
            per_sample_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "split": sample["split"],
                    "status": "computed",
                    "direct_alignment_rmse": metrics["direct_alignment_rmse"],
                    "max_abs_error": metrics["max_abs_error"],
                    "worker_id": result.worker_id,
                    "pid": result.pid,
                }
            )
            for k_index, value in enumerate(metrics["per_k_rmse"]):
                per_k_rows.append({"sample_id": sample["sample_id"], "k_index": k_index, "rmse": value, "mae": ""})
            for band_index, value in enumerate(metrics["per_band_rmse"]):
                per_band_rows.append({"sample_id": sample["sample_id"], "band_index": band_index, "rmse": value, "mae": ""})

        repeat_count = int(comparison_cfg.get("repeat_first_sample_count", 2))
        if sample_metrics and repeat_count >= 2:
            first_sample = selected_samples[0]
            first_idx = int(first_sample["dataset_index"])
            first_rho = np.asarray(data["rho"][first_idx], dtype=np.float64)
            first_recomputed = np.load(recomputed_dir / f"sample_{first_idx}_recomputed.npy")
            repeat, _ = solver_fn(first_rho, k_points=sidecar.k_points, config=solver)
            repeat_variation = _metric_block(repeat, first_recomputed)
            repeat_variation["status"] = "computed"

        if failures:
            decision = "solver_execution_failed"
        elif not sample_metrics:
            decision = "solver_execution_failed"
        else:
            max_abs = max(float(row["max_abs_error"]) for row in sample_metrics)
            if accepted_tolerance is None and repeat_variation.get("status") == "computed":
                accepted_tolerance = max(1.0e-7, 10.0 * float(repeat_variation["max_abs_error"]), float(solver.eigensolver_tolerance))
            if accepted_tolerance is None:
                decision = "insufficient_provenance"
            elif max_abs <= float(accepted_tolerance):
                decision = "physically_verified_with_documented_numerical_tolerance"
            else:
                decision = "mismatch_detected"

        sidecar_manifest_status, promoted_sidecar = _promote_verified_sidecar(
            cfg=cfg,
            dataset_cfg=dataset_cfg,
            dataset_path=dataset_path,
            sidecar_path=sidecar_path,
            manifest_path=manifest_path,
            run_dir=run_dir,
            decision=decision,
            accepted_tolerance=None if accepted_tolerance is None else float(accepted_tolerance),
        )
        metrics_payload = {
            "physical_verification_decision": decision,
            "real_mpb_execution_started": bool(sample_metrics or failures),
            "real_mpb_execution_completed": bool(len(sample_metrics) == len(selected_samples) and not failures),
            "direct_alignment_verified": decision in {"physically_verified", "physically_verified_with_documented_numerical_tolerance"},
            "direct_alignment_rmse": float(np.mean([row["direct_alignment_rmse"] for row in sample_metrics])) if sample_metrics else None,
            "direct_alignment_mae": float(np.mean([row["direct_alignment_mae"] for row in sample_metrics])) if sample_metrics else None,
            "max_abs_error": float(max([row["max_abs_error"] for row in sample_metrics])) if sample_metrics else None,
            "avg_MRE_bands_2_10_pct": float(np.mean([row["avg_MRE_bands_2_10_pct"] for row in sample_metrics])) if sample_metrics else None,
            "repeat_run_numerical_variation": repeat_variation,
            "accepted_tolerance": accepted_tolerance,
            "tolerance_rationale": "max(1e-7, 10x repeat first-sample max_abs, eigensolver tolerance)" if accepted_tolerance is not None else "not justified",
            "mpb_replay_time_s": float(sum(float(row["mpb_replay_time_s"]) for row in sample_metrics)),
            "wall_time_total_s": float(time.perf_counter() - wall_start),
            "sample_count": int(len(sample_metrics)),
            "sidecar_manifest_status": sidecar_manifest_status,
            "promoted_k_path_sidecar": promoted_sidecar,
            **worker_summary,
        }
        write_metrics(run_dir, metrics_payload)
        write_results(
            run_dir,
            {
                "action": "k_path_physical_validation_5sample",
                "status": decision,
                "formal_full_2a_allowed": decision in {"physically_verified", "physically_verified_with_documented_numerical_tolerance"},
                "samples": sample_metrics,
                "failures": failures,
                "sidecar_manifest_status": sidecar_manifest_status,
                "promoted_k_path_sidecar": promoted_sidecar,
            },
        )
        _write_csv(run_dir / "per_sample_metrics.csv", per_sample_rows, ["sample_id", "split", "status", "direct_alignment_rmse", "max_abs_error", "worker_id", "pid"])
        _write_csv(run_dir / "per_k_metrics.csv", per_k_rows, ["sample_id", "k_index", "rmse", "mae"])
        _write_csv(run_dir / "per_band_metrics.csv", per_band_rows, ["sample_id", "band_index", "rmse", "mae"])
        _write_csv(run_dir / "failure_cases.csv", failures, ["sample_id", "split", "status", "reason"])
        _write_json(run_dir / "alignment_diagnostics.json", diagnostics_payload)
        _write_json(run_dir / "selected_samples.json", {"samples": selected_samples})
        write_run_meta(
            run_dir,
            {
                "script": "scripts/run_k_path_physical_validation.py",
                "config_path": str(config_path),
                "validation_boundary": "wsl_meep_env_mpb_physical_validation",
                "status": decision,
            },
        )
        (run_dir / "history.csv").write_text(
            "stage,status,sample_count\nmpb_physical_validation," + decision + f",{len(sample_metrics)}\n",
            encoding="utf-8",
        )
        logger.info("Physical validation finished: %s", decision)
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Run five-sample WSL meep_env real-MPB k-path physical validation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--workers", default=None, help="MPB worker count: integer or auto.")
    parser.add_argument("--min-workers", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--worker-backend", default=None)
    args = parser.parse_args(argv)
    workers: int | str | None = None
    if args.workers is not None:
        workers = str(args.workers)
        if workers.lower() != "auto":
            workers = int(workers)
    run_dir = run_validation(
        args.config,
        run_dir_override=args.run_dir,
        workers=workers,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        worker_backend=args.worker_backend,
    )
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
