"""Generate a small real-MPB validation reference for B4 resolution auditing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.physics import canonicalize_complex_efield, evaluate_te_bloch_eigen_residual
from phc2d.runs import ensure_figures_dir, make_run_dir, setup_logging, write_json, write_metrics, write_resolved_config, write_results, write_run_meta
try:  # ``python scripts/...`` and ``pytest`` expose different import roots.
    from scripts.run_field_eigen_te_generation import ROOT, _file_hash, run_generation
except ModuleNotFoundError:  # pragma: no cover - exercised by the WSL script entry point.
    from run_field_eigen_te_generation import ROOT, _file_hash, run_generation


def _repo(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _selection(manifest: Mapping[str, Any], *, source_count: int) -> dict[str, Any]:
    records = list(manifest["records"])
    val_sources = sorted({int(row["source_sample_index"]) for row in records if str(row["source_split"]) == "val"}, key=lambda value: hashlib.sha256(f"b4-resolution-reference:{value}".encode()).hexdigest())[:source_count]
    if len(val_sources) != source_count:
        raise ValueError("B3 manifest has too few validation structures for the B4 reference audit.")
    selected = [dict(row) for row in records if int(row["source_sample_index"]) in set(val_sources) and str(row["source_split"]) == "val"]
    expected_per_source = len(selected) // source_count
    if not selected or any(sum(int(row["source_sample_index"]) == source for row in selected) != expected_per_source for source in val_sources):
        raise ValueError("Frozen B3 validation selection is not balanced by source structure.")
    payload = {
        "schema_version": "phc2d.field_eigen_selection.v2",
        "selection_id": f"b4_resolution_reference_val{source_count}_v1",
        "contract_id": manifest["contract_id"],
        "contract_version": manifest["contract_version"],
        "dataset_path": manifest["dataset_path"],
        "dataset_sha256": manifest["dataset_sha256"],
        "k_path_id": manifest["k_path_id"],
        "k_points_hash": manifest["k_points_hash"],
        "selector": {"seed": "sha256_b4_resolution_reference", "source_structure_split_counts": {"val": source_count}, "source_split_counts": {"val": len(selected)}, "k_per_structure": expected_per_source // len({int(row["band_index_one_based"]) for row in selected}), "band_range_one_based": manifest["selector"]["band_range_one_based"], "band_count_policy": "all_bands_per_structure_k", "solver_shard_unit": "structure_k_all_bands"},
        "records": selected,
    }
    payload["manifest_sha256"] = _canonical_hash(payload)
    return payload


def _residual_rows(field_path: Path, *, resolution: int) -> list[dict[str, Any]]:
    import h5py

    rows: list[dict[str, Any]] = []
    with h5py.File(field_path, "r") as handle:
        decode = lambda value: value.decode("utf-8") if isinstance(value, bytes) else str(value)
        epsilon = np.asarray(handle["epsilon"]); efield = np.asarray(handle["efield_complex"]); frequency = np.asarray(handle["eigenfrequency"]); k_points = np.asarray(handle["k_points"]); bands = np.asarray(handle["band_index"]); source_ids = np.asarray(handle["source_sample_id"]); dx = np.asarray(handle["dx"]); dy = np.asarray(handle["dy"])
    for index in range(len(frequency)):
        field = canonicalize_complex_efield(
            efield[index, ..., 0, :2], epsilon=epsilon[index], dx=float(dx[index]), dy=float(dy[index])
        )
        residual = evaluate_te_bloch_eigen_residual(field, epsilon=epsilon[index], eigenfrequency=float(frequency[index]), dx=float(dx[index]), dy=float(dy[index]), bloch_k_point=k_points[index])
        rows.append({"resolution": int(resolution), "source_sample_id": decode(source_ids[index]), "k_x": float(k_points[index, 0]), "k_y": float(k_points[index, 1]), "band_index_one_based": int(bands[index]), "eigenfrequency": float(frequency[index]), "maxwell_relative_residual": float(residual["relative_residual"]), "boundary_relative_residual": float(residual["boundary_relative_residual"]), "field_shape": "x".join(str(value) for value in field.shape[:2])})
    return rows


def run_audit(config_path: Path, *, run_dir_override: Path | None = None) -> Path:
    started = time.perf_counter()
    config_path = _repo(config_path); cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = dict(cfg["experiment"]); out = make_run_dir(str(experiment["experiment_id"]), runs_dir=_repo(dict(cfg.get("output") or {}).get("runs_dir", "runs")), run_dir=run_dir_override)
    logger = setup_logging(log_file=out / "run.log", run_meta={"script": "scripts/run_field_eigen_te_b4_resolution_reference_audit.py"}); ensure_figures_dir(out)
    manifest_path = _repo(str(cfg["b3_selection_manifest_path"])); b3_manifest = json.loads(manifest_path.read_text(encoding="utf-8")); selection = _selection(b3_manifest, source_count=int(cfg["selection"]["validation_source_count"]))
    write_json(out / "reference_selection_manifest.json", selection)
    child_runs: list[dict[str, Any]] = []; all_rows: list[dict[str, Any]] = []
    base_config = yaml.safe_load(_repo(str(cfg["generation_config_path"])).read_text(encoding="utf-8")) or {}
    for resolution in (int(value) for value in cfg["resolutions"]):
        child = {**base_config, "experiment": {"experiment_id": f"{experiment['experiment_id']}_r{resolution}"}, "selection_manifest_path": str(out / "reference_selection_manifest.json"), "selection": None, "reference_audit": True, "solver_override": {"resolution": resolution}, "output": {"runs_dir": str(out / "reference_runs"), "field_filename": f"reference_r{resolution}.h5"}}
        child_path = out / f"generation_r{resolution}.yaml"; child_path.write_text(yaml.safe_dump(child, sort_keys=False), encoding="utf-8")
        child_dir = out / "reference_runs" / f"r{resolution}"
        generated = run_generation(child_path, run_dir_override=child_dir)
        field_path = generated / f"reference_r{resolution}.h5"
        if not field_path.is_file():
            raise RuntimeError(f"Resolution reference generation did not write {field_path.name}.")
        child_runs.append({"resolution": resolution, "run_dir": str(generated), "field_path": str(field_path), "field_sha256": _file_hash(field_path)})
        all_rows.extend(_residual_rows(field_path, resolution=resolution))
        logger.info("Completed real-MPB reference resolution=%d", resolution)
    _write_csv(out / "resolution_reference_metrics.csv", all_rows)
    base_resolution = int(cfg["decision"]["base_resolution"]); limit = float(cfg["decision"]["residual_median_degradation_max_pct"])
    medians = {resolution: float(np.median([float(row["maxwell_relative_residual"]) for row in all_rows if int(row["resolution"]) == resolution])) for resolution in sorted({int(row["resolution"]) for row in all_rows})}
    changes = {str(resolution): 100.0 * (value / max(medians[base_resolution], 1.0e-12) - 1.0) for resolution, value in medians.items() if resolution != base_resolution}
    decision = {"status": "pass" if all(np.isfinite(list(medians.values()))) and all(value <= limit for value in changes.values()) else "blocked", "base_resolution": base_resolution, "residual_medians": medians, "residual_median_change_pct": changes, "residual_median_degradation_max_pct": limit, "meaning": "real_mpb_reference_only_not_a_pino_or_inverse_result"}
    wall_time_total_s = time.perf_counter() - started
    write_json(out / "reference_audit_decision.json", decision); write_resolved_config(out, yaml.safe_dump(cfg, sort_keys=False)); write_json(out / "lineage.json", {"b3_selection_manifest": str(manifest_path), "b3_selection_manifest_sha256": _file_hash(manifest_path), "reference_selection_manifest_sha256": _file_hash(out / "reference_selection_manifest.json"), "runs": child_runs}); write_metrics(out, {"record_count": len(all_rows), "real_mpb_reference_completed": True, "wall_time_total_s": wall_time_total_s, **decision}); write_results(out, {"status": "b4_resolution_reference_audit_completed", "decision": decision}); write_run_meta(out, {"status": "b4_resolution_reference_audit_completed", "validation_boundary": "wsl_meep_env_real_mpb", "wall_time_total_s": wall_time_total_s})
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Run B4 real-MPB resolution reference audit."); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args(argv); output = run_audit(args.config, run_dir_override=args.run_dir); print(output); return output


if __name__ == "__main__":
    main()
