"""Diagnose V2 smoke MPB top-band stability with guard-band solves.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from phc2d.data.k_path import load_k_path_spec
from phc2d.runs import ensure_figures_dir, make_run_dir, write_metrics, write_resolved_config, write_results, write_run_meta
from phc2d.solvers import MpbBandSolverConfig, solve_bands_for_rho_mpb


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAME = "scripts/run_v2_smoke_guard_band_diagnostic.py"


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


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _field_value(provenance: Mapping[str, Any], name: str, default: Any = None) -> Any:
    fields = provenance.get("fields")
    if isinstance(fields, Mapping) and name in fields:
        raw = fields[name]
        if isinstance(raw, Mapping) and "value" in raw:
            return raw["value"]
        return raw
    return provenance.get(name, default)


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _discover_dataset_path(generation_run: Path, dataset_path: Path | None) -> Path:
    if dataset_path is not None:
        return dataset_path
    results_path = generation_run / "results.json"
    if results_path.is_file():
        raw = _read_json(results_path).get("dataset_path")
        if raw:
            candidate = _resolve_path(str(raw))
            if candidate.is_file():
                return candidate
    candidates = sorted(generation_run.glob("*.npz"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Cannot discover one dataset NPZ under {generation_run}")


def _discover_sidecar_path(generation_run: Path, dataset_path: Path, sidecar_path: Path | None) -> Path:
    if sidecar_path is not None:
        return sidecar_path
    candidate = dataset_path.parent / "metadata" / f"{dataset_path.stem}.kpath.v1.json"
    if candidate.is_file():
        return candidate
    candidate = generation_run / "k_path_sidecar.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Cannot discover k-path sidecar for {dataset_path}")


def run_diagnostic(
    *,
    generation_run: str | Path,
    sample_index: int,
    k_indices: Sequence[int],
    guard_counts: Sequence[int],
    dataset_path: str | Path | None = None,
    sidecar_path: str | Path | None = None,
    run_dir_override: str | Path | None = None,
    solver_fn=solve_bands_for_rho_mpb,
) -> Path:
    wall_start = time.perf_counter()
    generation_run = _resolve_path(generation_run)
    dataset = _discover_dataset_path(generation_run, _resolve_path(dataset_path) if dataset_path is not None else None)
    sidecar = _discover_sidecar_path(generation_run, dataset, _resolve_path(sidecar_path) if sidecar_path is not None else None)
    provenance = _read_json(generation_run / "solver_provenance.json")
    requested_num_bands = int(_field_value(provenance, "num_bands"))
    base_cfg = MpbBandSolverConfig(
        num_bands=requested_num_bands,
        resolution=int(_field_value(provenance, "resolution")),
        polarization=str(_field_value(provenance, "polarization")),
        epsilon_air=float(_field_value(provenance, "epsilon_air")),
        epsilon_material=float(_field_value(provenance, "epsilon_material")),
        eigensolver_tolerance=float(_field_value(provenance, "eigensolver_tolerance")),
        grid_shape=tuple(int(v) for v in _field_value(provenance, "grid_shape")),  # type: ignore[arg-type]
        band_order_policy=str(_field_value(provenance, "band_order_policy", "frequency_ascending")),
    )
    spec = load_k_path_spec(sidecar)
    k_indices = [int(v) for v in k_indices]
    guard_counts = [int(v) for v in guard_counts]
    run_dir = make_run_dir("v2_smoke_guard_band_diagnostic", run_dir=run_dir_override)
    ensure_figures_dir(run_dir)
    write_run_meta(
        run_dir,
        {
            "script": SCRIPT_NAME,
            "generation_run": _display_path(generation_run),
            "dataset_path": _display_path(dataset),
            "sidecar_path": _display_path(sidecar),
            "sample_index": int(sample_index),
            "k_indices": k_indices,
            "guard_counts": guard_counts,
            "validation_boundary": "wsl_meep_env_real_mpb_guard_band_diagnostic",
        },
    )
    write_resolved_config(
        run_dir,
        "\n".join(
            [
                f"script: {SCRIPT_NAME}",
                f"generation_run: {_display_path(generation_run)}",
                f"dataset_path: {_display_path(dataset)}",
                f"sidecar_path: {_display_path(sidecar)}",
                f"sample_index: {int(sample_index)}",
                f"k_indices: {k_indices}",
                f"guard_counts: {guard_counts}",
                "",
            ]
        ),
    )
    with np.load(dataset, allow_pickle=False) as data:
        rho = np.asarray(data["rho"][int(sample_index)], dtype=np.float64)
        stored = np.asarray(data["Wh"][int(sample_index)], dtype=np.float64)
    k_points = np.asarray(spec.k_points, dtype=np.float64)[k_indices]
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    max_abs_by_guard: dict[int, float] = {}
    for guard_count in guard_counts:
        cfg = MpbBandSolverConfig(
            num_bands=base_cfg.num_bands,
            resolution=base_cfg.resolution,
            polarization=base_cfg.polarization,
            epsilon_air=base_cfg.epsilon_air,
            epsilon_material=base_cfg.epsilon_material,
            lattice_constant=base_cfg.lattice_constant,
            eigensolver_tolerance=base_cfg.eigensolver_tolerance,
            grid_shape=base_cfg.grid_shape,
            band_guard_count=max(0, int(guard_count)),
            band_order_policy=base_cfg.band_order_policy,
        )
        solve_start = time.perf_counter()
        bands, meta = solver_fn(rho, k_points=k_points, config=cfg)
        solve_s = float(time.perf_counter() - solve_start)
        bands = np.asarray(bands, dtype=np.float64)
        np.save(run_dir / f"sample_{int(sample_index)}_guard_{int(guard_count)}_bands.npy", bands)
        abs_errors = []
        for local_pos, k_index in enumerate(k_indices):
            for band_index in range(requested_num_bands):
                stored_value = float(stored[int(k_index), band_index])
                recomputed_value = float(bands[local_pos, band_index])
                abs_error = abs(recomputed_value - stored_value)
                abs_errors.append(abs_error)
                rows.append(
                    {
                        "guard_count": int(guard_count),
                        "sample_index": int(sample_index),
                        "k_index": int(k_index),
                        "band_index": int(band_index),
                        "stored": stored_value,
                        "recomputed": recomputed_value,
                        "delta": recomputed_value - stored_value,
                        "abs_error": abs_error,
                    }
                )
        max_abs = float(max(abs_errors)) if abs_errors else 0.0
        max_abs_by_guard[int(guard_count)] = max_abs
        run_rows.append(
            {
                "guard_count": int(guard_count),
                "solve_time_s": solve_s,
                "requested_num_bands": int(meta.get("num_bands", requested_num_bands)),
                "solved_num_bands": int(meta.get("solved_num_bands", requested_num_bands + max(0, int(guard_count)))),
                "max_abs_vs_stored_on_selected_k": max_abs,
            }
        )
    _write_csv(
        run_dir / "guard_band_comparison.csv",
        rows,
        ["guard_count", "sample_index", "k_index", "band_index", "stored", "recomputed", "delta", "abs_error"],
    )
    _write_csv(
        run_dir / "guard_band_runs.csv",
        run_rows,
        ["guard_count", "solve_time_s", "requested_num_bands", "solved_num_bands", "max_abs_vs_stored_on_selected_k"],
    )
    root_cause = (
        "top_retained_band_guard_missing"
        if any(count > 0 for count in guard_counts) and len({round(v, 12) for v in max_abs_by_guard.values()}) > 1
        else "guard_band_effect_not_detected"
    )
    summary = {
        "status": "guard_band_diagnostic_completed",
        "root_cause_candidate": root_cause,
        "sample_index": int(sample_index),
        "k_indices": k_indices,
        "guard_counts": guard_counts,
        "max_abs_by_guard": max_abs_by_guard,
        "interpretation": (
            "Changing only the number of solved guard bands changes the top retained band comparison, "
            "which indicates an under-guarded highest retained band near a competing branch."
            if root_cause == "top_retained_band_guard_missing"
            else "No guard-band effect was detected on the selected k-points."
        ),
        "wall_time_total_s": float(time.perf_counter() - wall_start),
    }
    _write_json(run_dir / "guard_band_summary.json", summary)
    write_metrics(
        run_dir,
        {
            "diagnostic_status": summary["status"],
            "root_cause_candidate": root_cause,
            "sample_index": int(sample_index),
            "selected_k_count": len(k_indices),
            "guard_count_count": len(guard_counts),
            "wall_time_total_s": summary["wall_time_total_s"],
        },
    )
    write_results(
        run_dir,
        {
            "action": "v2_smoke_guard_band_diagnostic",
            "status": summary["status"],
            "root_cause_candidate": root_cause,
            "formal_full_2a_allowed": False,
        },
    )
    (run_dir / "history.csv").write_text(
        "stage,status,reason\n"
        f"guard_band_diagnostic,{summary['status']},{root_cause}\n",
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text(
        f"Guard-band diagnostic status: {summary['status']}\n"
        f"Root-cause candidate: {root_cause}\n"
        f"Max abs by guard: {max_abs_by_guard}\n",
        encoding="utf-8",
    )
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Run targeted real-MPB guard-band diagnostics for V2 smoke samples.")
    parser.add_argument("--generation-run", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--sidecar-path", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=3)
    parser.add_argument("--k-indices", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--guard-counts", type=str, default="0,4")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    run_dir = run_diagnostic(
        generation_run=args.generation_run,
        dataset_path=args.dataset_path,
        sidecar_path=args.sidecar_path,
        sample_index=args.sample_index,
        k_indices=_parse_int_list(args.k_indices),
        guard_counts=_parse_int_list(args.guard_counts),
        run_dir_override=args.run_dir,
    )
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
