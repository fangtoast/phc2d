"""Audit the B2 TE Bloch-eigenfield residual gate without training a model.

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
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.physics import evaluate_te_bloch_eigen_residual, make_deterministic_random_efield
from phc2d.runs import ensure_figures_dir, make_run_dir, setup_logging, write_json, write_metrics, write_resolved_config, write_results, write_run_meta


ROOT = Path(__file__).resolve().parents[1]


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return payload


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def run_residual_gate(config_path: Path, *, run_dir_override: Path | None = None) -> Path:
    """Evaluate solver references against normalized deterministic random fields."""

    config_path = _repo_path(config_path)
    cfg = _load_yaml(config_path)
    experiment_id = str(dict(cfg["experiment"])["experiment_id"])
    run_dir = make_run_dir(experiment_id, runs_dir=_repo_path(dict(cfg.get("output") or {}).get("runs_dir", "runs")), run_dir=run_dir_override)
    logger = setup_logging(log_file=run_dir / "run.log", run_meta={"script": "scripts/run_field_eigen_te_residual_gate.py", "config_path": str(config_path)})
    ensure_figures_dir(run_dir)
    started = time.perf_counter()
    contract_path = _repo_path(str(cfg["contract_path"]))
    contract = _load_yaml(contract_path)
    if str(contract.get("contract", {}).get("id")) != "field_eigen_te_contract":
        raise ValueError("Residual gate requires the frozen TE field contract.")
    required_fields = set(contract["record_schema"]["required_fields"])
    required_provenance = set(contract["record_schema"]["solver_provenance_required"])
    dataset_path = _repo_path(str(cfg["field_dataset_path"]))
    source_manifest_path = _repo_path(str(cfg["source_manifest_path"]))
    source_manifest = _load_yaml(source_manifest_path)
    if str(source_manifest.get("dataset_file_hash")) != _file_hash(dataset_path):
        raise ValueError("B2 input HDF5 hash does not match its B1 dataset manifest.")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ImportError("h5py is required to audit field residuals.") from exc
    schema_failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    with h5py.File(dataset_path, "r") as handle:
        datasets = set(handle.keys())
        aliases = {"boundary_condition", "field_normalization", "solver_provenance", "domain_extent", "dx", "dy"}
        # The B1 schema uses storage-compatible plural/index names while the
        # frozen contract names their per-record semantic counterparts.
        serialized_aliases = {"band_index_one_based": "band_index", "k_point": "k_points"}
        available_contract_fields = datasets | aliases | {name for name, dataset_name in serialized_aliases.items() if dataset_name in datasets}
        missing = sorted(required_fields - available_contract_fields)
        if missing:
            schema_failures.append({"scope": "dataset", "missing": missing})
        metadata = json.loads(_decode(handle.attrs.get("metadata_json", "{}")))
        if str(handle.attrs.get("boundary_condition", "")) != "bloch_periodic" or str(metadata.get("boundary_condition", "")) != "bloch_periodic":
            schema_failures.append({"scope": "dataset", "missing": ["bloch_periodic_boundary_condition"]})
        count = int(handle["efield_complex"].shape[0])
        for index in range(count):
            record_id = _decode(handle["record_id"][index])
            provenance = json.loads(_decode(handle["solver_provenance_json"][index]))
            missing_provenance = sorted(required_provenance - set(provenance))
            if missing_provenance:
                schema_failures.append({"record_id": record_id, "missing": missing_provenance})
                continue
            efield = np.asarray(handle["efield_complex"][index])
            epsilon = np.asarray(handle["epsilon"][index])
            kwargs = {
                "epsilon": epsilon,
                "eigenfrequency": float(handle["eigenfrequency"][index]),
                "dx": float(handle["dx"][index]),
                "dy": float(handle["dy"][index]),
                "bloch_k_point": np.asarray(handle["bloch_phase"][index]),
            }
            reference = evaluate_te_bloch_eigen_residual(efield, **kwargs)
            random_field = make_deterministic_random_efield(
                efield.shape, record_id=record_id, epsilon=kwargs["epsilon"], dx=kwargs["dx"], dy=kwargs["dy"]
            )
            random_result = evaluate_te_bloch_eigen_residual(random_field, **kwargs)
            rows.append({
                "record_id": record_id,
                "source_split": _decode(handle["source_split"][index]),
                "reference_relative_residual": reference["relative_residual"],
                "random_relative_residual": random_result["relative_residual"],
                "reference_boundary_relative_residual": reference["boundary_relative_residual"],
                "random_boundary_relative_residual": random_result["boundary_relative_residual"],
                "solver_beats_random": bool(reference["relative_residual"] < random_result["relative_residual"]),
                "epsilon_upsample_ratio": reference["epsilon_upsample_ratio"],
            })
            logger.info("Residual audited %s", record_id)
    gate_cfg = dict(cfg.get("gate") or {})
    reference_values = np.asarray([row["reference_relative_residual"] for row in rows], dtype=np.float64)
    random_values = np.asarray([row["random_relative_residual"] for row in rows], dtype=np.float64)
    win_rate = float(np.mean([row["solver_beats_random"] for row in rows])) if rows else 0.0
    solver_median = float(np.median(reference_values)) if len(reference_values) else float("inf")
    random_median = float(np.median(random_values)) if len(random_values) else float("inf")
    passed = not schema_failures and bool(rows) and win_rate >= float(gate_cfg["minimum_solver_win_rate"]) and (not bool(gate_cfg["require_solver_median_below_random"]) or solver_median < random_median)
    status = "b2_residual_gate_pass" if passed else "b2_residual_gate_blocked"
    with (run_dir / "case_residuals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "source_split", "reference_relative_residual", "random_relative_residual", "reference_boundary_relative_residual", "random_boundary_relative_residual", "solver_beats_random", "epsilon_upsample_ratio"])
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "status": status,
        "dataset_path": str(dataset_path.relative_to(ROOT)),
        "dataset_sha256": _file_hash(dataset_path),
        "source_manifest_path": str(source_manifest_path.relative_to(ROOT)),
        "source_manifest_sha256": _file_hash(source_manifest_path),
        "record_count": len(rows),
        "schema_failures": schema_failures,
        "solver_reference_median_relative_residual": solver_median,
        "random_field_median_relative_residual": random_median,
        "solver_reference_win_rate": win_rate,
        "minimum_solver_win_rate": float(gate_cfg["minimum_solver_win_rate"]),
        "solver_reference_beats_random_field": bool(solver_median < random_median and win_rate >= float(gate_cfg["minimum_solver_win_rate"])),
        "field_model_training_started": False,
        "pino_started": False,
        "maxwellnet_started": False,
    }
    write_json(run_dir / "residual_audit.json", audit)
    write_resolved_config(run_dir, yaml.safe_dump({"residual_gate": cfg, "contract": contract}, sort_keys=False, allow_unicode=True))
    write_metrics(run_dir, {**audit, "wall_time_total_s": float(time.perf_counter() - started)})
    write_results(run_dir, {"action": "field_eigen_te_b2_residual_gate", "status": status, "residual_audit": "residual_audit.json", "next_gate": "B1_1000_real_mpb_generation" if passed else "B2_residual_formula_or_lineage_remediation"})
    write_run_meta(run_dir, {"script": "scripts/run_field_eigen_te_residual_gate.py", "config_path": str(config_path.relative_to(ROOT)), "validation_boundary": "wsl_meep_env_real_mpb_reference_residual", "status": status})
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Audit TE Bloch eigenfield residuals against deterministic random fields.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_residual_gate(args.config, run_dir_override=args.run_dir)
    print(result)
    return result


if __name__ == "__main__":
    main()
