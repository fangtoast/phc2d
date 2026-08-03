"""Audit Query Dataset V2 smoke replay mismatches without re-running MPB.

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
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from phc2d.data.forward_dataset import parse_metadata_jsonl
from phc2d.data.lift import lift_rho_to_grid
from phc2d.runs import ensure_figures_dir, make_run_dir, write_metrics, write_resolved_config, write_results, write_run_meta


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAME = "scripts/run_v2_smoke_mismatch_audit.py"


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


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _jsonable(row.get(name, "")) for name in fieldnames})
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a JSON object at {path}")
    return dict(payload)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _generation_hash_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    header = json.dumps({"shape": list(arr.shape), "dtype": str(arr.dtype)}, sort_keys=True).encode("utf-8")
    return _sha256_bytes(header + b"\0" + arr.tobytes())


def _plain_hash_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    return _sha256_bytes(arr.tobytes())


def _as_field_value(payload: Mapping[str, Any], name: str) -> Any:
    fields = payload.get("fields")
    if isinstance(fields, Mapping) and name in fields:
        raw = fields[name]
        if isinstance(raw, Mapping) and "value" in raw:
            return raw["value"]
        return raw
    return payload.get(name)


def _find_dataset_path(generation_run: Path, dataset_path: Path | None) -> Path:
    if dataset_path is not None:
        return dataset_path
    results_path = generation_run / "results.json"
    if results_path.is_file():
        payload = _read_json(results_path)
        raw = payload.get("dataset_path")
        if raw:
            candidate = _resolve_path(str(raw))
            if candidate.is_file():
                return candidate
    candidates = sorted(generation_run.glob("*.npz"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No NPZ dataset found under {generation_run}")
    raise ValueError(f"Multiple NPZ datasets found under {generation_run}; pass --dataset-path explicitly")


def _grid_shape_from_provenance(validation_run: Path, dataset: Mapping[str, np.ndarray]) -> tuple[int, int]:
    provenance_path = validation_run / "solver_provenance.json"
    if provenance_path.is_file():
        value = _as_field_value(_read_json(provenance_path), "grid_shape")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
    rho_len = int(np.asarray(dataset["rho"]).shape[1])
    # Let the lifting helper raise a shape-specific error if the length is not
    # compatible with a p4m wedge; this keeps the audit shape-derived.
    return lift_rho_to_grid(np.zeros(rho_len, dtype=np.float32)).shape  # type: ignore[return-value]


def _load_sample_identity_rows(
    *,
    validation_run: Path,
    generation_run: Path,
    dataset_path: Path,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    validation_rows = _read_csv(validation_run / "selected_samples.csv")
    generation_rows_by_id: dict[str, dict[str, Any]] = {}
    generation_metadata_path = generation_run / "sample_metadata.jsonl"
    if generation_metadata_path.is_file():
        for line in generation_metadata_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, Mapping):
                    generation_rows_by_id[str(row.get("sample_id"))] = dict(row)

    rows: list[dict[str, Any]] = []
    metadata_by_index: dict[int, dict[str, Any]] = {}
    with np.load(dataset_path, allow_pickle=False) as data:
        metadata_rows = parse_metadata_jsonl(data["metadata_jsonl"]) if "metadata_jsonl" in data else []
        grid_shape = _grid_shape_from_provenance(validation_run, data)
        for row in validation_rows:
            dataset_index = int(row["dataset_index"])
            sample_id = str(row["sample_id"])
            npz_meta = metadata_rows[dataset_index] if dataset_index < len(metadata_rows) else {}
            gen_meta = generation_rows_by_id.get(sample_id, {})
            rho = np.asarray(data["rho"][dataset_index])
            wh = np.asarray(data["Wh"][dataset_index])
            grid = (lift_rho_to_grid(rho, grid_shape=grid_shape) > 0.5).astype(np.uint8)
            computed = {
                "rho_hash": _generation_hash_array(rho),
                "geometry_hash": _generation_hash_array(grid),
                "Wh_hash": _generation_hash_array(wh),
                "plain_rho_hash": _plain_hash_array(rho),
                "plain_geometry_hash": _plain_hash_array(grid),
                "plain_Wh_hash": _plain_hash_array(wh),
            }
            checks = {
                "rho_hash_match": row.get("rho_hash") == npz_meta.get("rho_hash") == gen_meta.get("rho_hash") == computed["rho_hash"],
                "geometry_hash_match": row.get("geometry_hash")
                == npz_meta.get("geometry_hash")
                == gen_meta.get("geometry_hash")
                == computed["geometry_hash"],
                "Wh_hash_match": row.get("existing_Wh_hash") == npz_meta.get("Wh_hash") == gen_meta.get("Wh_hash") == computed["Wh_hash"],
            }
            audit_row = {
                "sample_id": sample_id,
                "dataset_index": dataset_index,
                "split": row.get("split", ""),
                "rho_hash_match": checks["rho_hash_match"],
                "geometry_hash_match": checks["geometry_hash_match"],
                "Wh_hash_match": checks["Wh_hash_match"],
                "validation_rho_hash": row.get("rho_hash", ""),
                "npz_rho_hash": npz_meta.get("rho_hash", ""),
                "generation_rho_hash": gen_meta.get("rho_hash", ""),
                "computed_rho_hash": computed["rho_hash"],
                "validation_geometry_hash": row.get("geometry_hash", ""),
                "npz_geometry_hash": npz_meta.get("geometry_hash", ""),
                "generation_geometry_hash": gen_meta.get("geometry_hash", ""),
                "computed_geometry_hash": computed["geometry_hash"],
                "validation_Wh_hash": row.get("existing_Wh_hash", ""),
                "npz_Wh_hash": npz_meta.get("Wh_hash", ""),
                "generation_Wh_hash": gen_meta.get("Wh_hash", ""),
                "computed_Wh_hash": computed["Wh_hash"],
            }
            rows.append(audit_row)
            metadata_by_index[dataset_index] = {
                "validation": row,
                "npz": dict(npz_meta),
                "generation": gen_meta,
                "computed": computed,
                "checks": checks,
            }
    return rows, metadata_by_index


def _load_band_pairs(validation_run: Path) -> list[tuple[int, np.ndarray, np.ndarray]]:
    pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for stored_path in sorted((validation_run / "stored_bands").glob("sample_*_stored.npy")):
        stem = stored_path.name.removeprefix("sample_").removesuffix("_stored.npy")
        dataset_index = int(stem)
        recomputed_path = validation_run / "recomputed_bands" / f"sample_{dataset_index}_recomputed.npy"
        if not recomputed_path.is_file():
            raise FileNotFoundError(f"Missing recomputed pair for {stored_path}")
        pairs.append((dataset_index, np.load(stored_path), np.load(recomputed_path)))
    if not pairs:
        raise FileNotFoundError(f"No stored band arrays found under {validation_run / 'stored_bands'}")
    return pairs


def _metric_block(recomputed: np.ndarray, stored: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(recomputed, dtype=np.float64) - np.asarray(stored, dtype=np.float64)
    abs_delta = np.abs(delta)
    arg = np.unravel_index(int(np.argmax(abs_delta)), abs_delta.shape)
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae": float(np.mean(abs_delta)),
        "max_abs_error": float(abs_delta[arg]),
        "max_abs_k_index": int(arg[0]),
        "max_abs_band_index": int(arg[1]),
        "max_abs_delta": float(delta[arg]),
        "max_abs_stored": float(stored[arg]),
        "max_abs_recomputed": float(recomputed[arg]),
    }


def _mismatch_rows(
    *,
    validation_run: Path,
    metadata_by_index: Mapping[int, Mapping[str, Any]],
    significant_error_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    per_cell_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    significant_by_sample: dict[str, int] = defaultdict(int)
    significant_by_band: dict[int, int] = defaultdict(int)
    significant_by_k: dict[int, int] = defaultdict(int)

    for dataset_index, stored, recomputed in _load_band_pairs(validation_run):
        if stored.shape != recomputed.shape:
            raise ValueError(f"Stored/recomputed shape mismatch for dataset_index={dataset_index}: {stored.shape} vs {recomputed.shape}")
        meta = metadata_by_index.get(dataset_index, {})
        sample_id = str(dict(meta.get("validation", {})).get("sample_id", dataset_index))
        sample_metrics = _metric_block(recomputed, stored)
        delta = np.asarray(recomputed, dtype=np.float64) - np.asarray(stored, dtype=np.float64)
        abs_delta = np.abs(delta)
        significant_count = int(np.sum(abs_delta > float(significant_error_threshold)))
        per_sample_rows.append(
            {
                "sample_id": sample_id,
                "dataset_index": dataset_index,
                **sample_metrics,
                "significant_error_threshold": float(significant_error_threshold),
                "significant_cell_count": significant_count,
            }
        )
        for k_index in range(abs_delta.shape[0]):
            for band_index in range(abs_delta.shape[1]):
                value = float(abs_delta[k_index, band_index])
                is_significant = value > float(significant_error_threshold)
                if is_significant:
                    significant_by_sample[sample_id] += 1
                    significant_by_band[int(band_index)] += 1
                    significant_by_k[int(k_index)] += 1
                per_cell_rows.append(
                    {
                        "sample_id": sample_id,
                        "dataset_index": dataset_index,
                        "k_index": k_index,
                        "band_index": band_index,
                        "stored": float(stored[k_index, band_index]),
                        "recomputed": float(recomputed[k_index, band_index]),
                        "delta": float(delta[k_index, band_index]),
                        "abs_error": value,
                        "significant": is_significant,
                    }
                )

    worst = max(per_sample_rows, key=lambda row: float(row["max_abs_error"]))
    significant_cells = [row for row in per_cell_rows if row["significant"]]
    aggregate = {
        "sample_count": len(per_sample_rows),
        "cell_count": len(per_cell_rows),
        "significant_cell_count": len(significant_cells),
        "significant_samples": sorted(significant_by_sample),
        "significant_bands": sorted(int(k) for k in significant_by_band),
        "significant_k_indices": sorted(int(k) for k in significant_by_k),
        "worst_sample_id": worst["sample_id"],
        "worst_dataset_index": int(worst["dataset_index"]),
        "max_abs_error": float(worst["max_abs_error"]),
        "max_abs_k_index": int(worst["max_abs_k_index"]),
        "max_abs_band_index": int(worst["max_abs_band_index"]),
    }
    return per_cell_rows, per_sample_rows, aggregate


def _provenance_audit(validation_run: Path, generation_run: Path) -> dict[str, Any]:
    validation_path = validation_run / "solver_provenance.json"
    generation_path = generation_run / "solver_provenance.json"
    validation = _read_json(validation_path)
    generation = _read_json(generation_path)
    comparable_fields = (
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
        "frequency_normalization",
    )
    field_rows = []
    for name in comparable_fields:
        generation_value = _as_field_value(generation, name)
        validation_value = _as_field_value(validation, name)
        field_rows.append(
            {
                "field": name,
                "generation_value": generation_value,
                "validation_value": validation_value,
                "match": generation_value == validation_value,
            }
        )
    generation_k = _as_field_value(generation, "k_points")
    validation_k = _as_field_value(validation, "k_path")
    generation_k_hash = dict(generation_k or {}).get("hash") if isinstance(generation_k, Mapping) else None
    validation_k_hash = dict(validation_k or {}).get("k_points_hash") if isinstance(validation_k, Mapping) else None
    field_rows.append(
        {
            "field": "k_points_hash",
            "generation_value": generation_k_hash,
            "validation_value": validation_k_hash,
            "match": generation_k_hash == validation_k_hash,
        }
    )
    missing = list(validation.get("missing_critical_fields", [])) + list(generation.get("missing_required_fields", []))
    return {
        "generation_provenance_path": _display_path(generation_path),
        "validation_provenance_path": _display_path(validation_path),
        "generation_use_proxy": bool(generation.get("use_proxy", True)),
        "validation_use_proxy": bool(validation.get("use_proxy", True)),
        "missing_fields": missing,
        "field_comparisons": field_rows,
        "provenance_matches": bool(not missing and all(bool(row["match"]) for row in field_rows)),
    }


def _classify_root_cause(
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    mismatch: Mapping[str, Any],
) -> dict[str, Any]:
    identity_ok = all(bool(row["rho_hash_match"] and row["geometry_hash_match"] and row["Wh_hash_match"]) for row in identity_rows)
    provenance_ok = bool(provenance.get("provenance_matches"))
    significant_samples = list(mismatch.get("significant_samples", []))
    significant_bands = list(mismatch.get("significant_bands", []))
    significant_k_indices = list(mismatch.get("significant_k_indices", []))
    if identity_ok and provenance_ok and int(mismatch.get("significant_cell_count", 0)) == 0:
        return {
            "status": "fixed_and_v2_smoke_validation_passed",
            "root_cause_family": "no_significant_direct_alignment_mismatch",
            "root_cause": "Sample identity and solver provenance match, and no direct-alignment cell exceeds the significant mismatch threshold.",
            "formal_full_2a_allowed": True,
        }
    localized_high_band = (
        identity_ok
        and provenance_ok
        and len(significant_samples) == 1
        and len(significant_bands) == 1
        and int(mismatch.get("significant_cell_count", 0)) <= max(2, len(significant_k_indices))
    )
    if localized_high_band:
        return {
            "status": "mismatch_root_cause_identified",
            "root_cause_family": "localized_generation_stored_Wh_high_band_replay_divergence",
            "root_cause": (
                "Sample identity and solver provenance match; the direct-alignment failure is isolated to "
                f"sample {significant_samples[0]}, band index {significant_bands[0]}, k indices {significant_k_indices}. "
                "This rules out dataset selection, sidecar, global k-path, and solver-provenance mismatch as the cause."
            ),
            "formal_full_2a_allowed": False,
        }
    if not identity_ok:
        return {
            "status": "mismatch_root_cause_identified",
            "root_cause_family": "sample_identity_or_dataset_association_mismatch",
            "root_cause": "At least one selected sample hash differs across validation metadata, NPZ metadata, generation metadata, or recomputed NPZ arrays.",
            "formal_full_2a_allowed": False,
        }
    if not provenance_ok:
        return {
            "status": "mismatch_root_cause_identified",
            "root_cause_family": "solver_provenance_mismatch",
            "root_cause": "Generation and validation solver provenance fields do not match.",
            "formal_full_2a_allowed": False,
        }
    return {
        "status": "mismatch_unresolved_blocked",
        "root_cause_family": "unresolved_nonlocalized_replay_mismatch",
        "root_cause": "Identity and provenance checks pass, but the mismatch pattern is not localized enough for a minimal root-cause classification.",
        "formal_full_2a_allowed": False,
    }


def run_audit(
    *,
    validation_run: str | Path,
    generation_run: str | Path,
    dataset_path: str | Path | None = None,
    run_dir_override: str | Path | None = None,
    significant_error_threshold: float = 1.0e-4,
) -> Path:
    wall_start = time.perf_counter()
    validation_run = _resolve_path(validation_run)
    generation_run = _resolve_path(generation_run)
    dataset = _find_dataset_path(generation_run, _resolve_path(dataset_path) if dataset_path is not None else None)
    run_dir = make_run_dir("v2_smoke_mismatch_audit", run_dir=run_dir_override)
    ensure_figures_dir(run_dir)
    write_run_meta(
        run_dir,
        {
            "script": SCRIPT_NAME,
            "validation_run": _display_path(validation_run),
            "generation_run": _display_path(generation_run),
            "dataset_path": _display_path(dataset),
            "validation_boundary": "artifact_audit_only_no_mpb_execution",
        },
    )
    config_text = "\n".join(
        [
            f"script: {SCRIPT_NAME}",
            f"validation_run: {_display_path(validation_run)}",
            f"generation_run: {_display_path(generation_run)}",
            f"dataset_path: {_display_path(dataset)}",
            f"significant_error_threshold: {float(significant_error_threshold)}",
            "mpb_execution: false",
            "",
        ]
    )
    write_resolved_config(run_dir, config_text)

    identity_rows, metadata_by_index = _load_sample_identity_rows(
        validation_run=validation_run,
        generation_run=generation_run,
        dataset_path=dataset,
    )
    per_cell_rows, per_sample_rows, mismatch = _mismatch_rows(
        validation_run=validation_run,
        metadata_by_index=metadata_by_index,
        significant_error_threshold=float(significant_error_threshold),
    )
    provenance = _provenance_audit(validation_run, generation_run)
    classification = _classify_root_cause(identity_rows=identity_rows, provenance=provenance, mismatch=mismatch)

    _write_csv(
        run_dir / "sample_identity_audit.csv",
        identity_rows,
        [
            "sample_id",
            "dataset_index",
            "split",
            "rho_hash_match",
            "geometry_hash_match",
            "Wh_hash_match",
            "validation_rho_hash",
            "npz_rho_hash",
            "generation_rho_hash",
            "computed_rho_hash",
            "validation_geometry_hash",
            "npz_geometry_hash",
            "generation_geometry_hash",
            "computed_geometry_hash",
            "validation_Wh_hash",
            "npz_Wh_hash",
            "generation_Wh_hash",
            "computed_Wh_hash",
        ],
    )
    _write_csv(
        run_dir / "mismatch_per_sample.csv",
        per_sample_rows,
        [
            "sample_id",
            "dataset_index",
            "rmse",
            "mae",
            "max_abs_error",
            "max_abs_k_index",
            "max_abs_band_index",
            "max_abs_delta",
            "max_abs_stored",
            "max_abs_recomputed",
            "significant_error_threshold",
            "significant_cell_count",
        ],
    )
    _write_csv(
        run_dir / "mismatch_per_k_band.csv",
        per_cell_rows,
        ["sample_id", "dataset_index", "k_index", "band_index", "stored", "recomputed", "delta", "abs_error", "significant"],
    )
    _write_json(run_dir / "provenance_audit.json", provenance)

    wall_time = float(time.perf_counter() - wall_start)
    summary = {
        "status": classification["status"],
        "root_cause_family": classification["root_cause_family"],
        "root_cause": classification["root_cause"],
        "formal_full_2a_allowed": bool(classification["formal_full_2a_allowed"]),
        "dataset_path": _display_path(dataset),
        "generation_run": _display_path(generation_run),
        "validation_run": _display_path(validation_run),
        "identity_all_match": all(
            bool(row["rho_hash_match"] and row["geometry_hash_match"] and row["Wh_hash_match"]) for row in identity_rows
        ),
        "provenance_matches": bool(provenance.get("provenance_matches")),
        "mismatch": mismatch,
        "wall_time_total_s": wall_time,
    }
    _write_json(run_dir / "mismatch_summary.json", summary)
    write_metrics(
        run_dir,
        {
            "audit_status": classification["status"],
            "sample_count": int(mismatch["sample_count"]),
            "cell_count": int(mismatch["cell_count"]),
            "significant_cell_count": int(mismatch["significant_cell_count"]),
            "max_abs_error": float(mismatch["max_abs_error"]),
            "max_abs_k_index": int(mismatch["max_abs_k_index"]),
            "max_abs_band_index": int(mismatch["max_abs_band_index"]),
            "identity_all_match": bool(summary["identity_all_match"]),
            "provenance_matches": bool(summary["provenance_matches"]),
            "mpb_execution_started": False,
            "mpb_execution_completed": False,
            "formal_full_2a_allowed": bool(summary["formal_full_2a_allowed"]),
            "wall_time_total_s": wall_time,
        },
    )
    write_results(
        run_dir,
        {
            "action": "query_dataset_v2_smoke_mismatch_audit",
            "status": classification["status"],
            "root_cause_family": classification["root_cause_family"],
            "formal_full_2a_allowed": bool(classification["formal_full_2a_allowed"]),
            "generated_artifacts": [
                "mismatch_summary.json",
                "mismatch_per_sample.csv",
                "mismatch_per_k_band.csv",
                "sample_identity_audit.csv",
                "provenance_audit.json",
            ],
        },
    )
    (run_dir / "history.csv").write_text(
        "stage,status,reason\n"
        f"artifact_identity_check,{'passed' if summary['identity_all_match'] else 'failed'},\n"
        f"solver_provenance_check,{'passed' if summary['provenance_matches'] else 'failed'},\n"
        f"mismatch_localization,{classification['status']},{classification['root_cause_family']}\n",
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text(
        f"Audit status: {classification['status']}\n"
        f"Root cause family: {classification['root_cause_family']}\n"
        f"Max abs error: {mismatch['max_abs_error']} at sample {mismatch['worst_sample_id']} "
        f"k={mismatch['max_abs_k_index']} band={mismatch['max_abs_band_index']}\n",
        encoding="utf-8",
    )
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Audit V2 smoke direct-alignment mismatch artifacts without MPB execution.")
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--generation-run", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--significant-error-threshold", type=float, default=1.0e-4)
    args = parser.parse_args(argv)
    run_dir = run_audit(
        validation_run=args.validation_run,
        generation_run=args.generation_run,
        dataset_path=args.dataset_path,
        run_dir_override=args.run_dir,
        significant_error_threshold=args.significant_error_threshold,
    )
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
