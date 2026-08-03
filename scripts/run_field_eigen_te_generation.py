"""Generate contract-complete TE Bloch eigenfield records with real MPB.

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
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.data import (
    FieldEigenRecord,
    build_field_eigen_operator_selection_manifest,
    build_field_eigen_selection_manifest,
    canonicalize_eigenfields,
    epsilon_from_rho,
    hash_array,
    inspect_field_dataset,
    load_dataset_k_path_spec,
    validate_verified_k_path_reference,
    validate_field_eigen_selection_manifest,
    write_field_eigen_hdf5,
)
from phc2d.data.k_path import VERIFIED_K_PATH_STATUSES, hash_k_path_spec
from phc2d.runs import (
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    write_json,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)
from phc2d.solvers import (
    MPBParallelConfig,
    MPBWorkerJob,
    MpbBandSolverConfig,
    run_mpb_worker_jobs,
    solve_eigenfield_bundle_for_rho_mpb,
    solve_eigenfield_multiband_bundle_for_rho_mpb,
)


ROOT = Path(__file__).resolve().parents[1]


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_history(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = ["stage", "status", "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_metadata(raw: str, expected_count: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != expected_count:
        raise ValueError(f"metadata_jsonl contains {len(rows)} rows, expected {expected_count}")
    return [dict(row) for row in rows]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"YAML config must be a mapping: {path}")
    return dict(payload)


def _load_selection_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Field selection manifest must be a mapping: {path}")
    problems = validate_field_eigen_selection_manifest(payload)
    if problems:
        raise ValueError("Invalid field selection manifest: " + "; ".join(problems))
    return dict(payload)


def _source_split_map(bundle: Mapping[str, Any], count: int) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for name, field_name in (("train", "idx_train"), ("val", "idx_val"), ("test", "idx_test")):
        if field_name not in bundle:
            raise ValueError(f"Source dataset is missing {field_name} required for field lineage.")
        for value in np.asarray(bundle[field_name], dtype=np.int64).reshape(-1):
            index = int(value)
            if index in mapping:
                raise ValueError(f"Source split index {index} appears more than once.")
            mapping[index] = name
    if set(mapping) != set(range(int(count))):
        raise ValueError("Source train/val/test indices must partition every source sample.")
    return mapping


def run_generation(
    config_path: Path,
    *,
    run_dir_override: Path | None = None,
    b2_gate_run_dir: Path | None = None,
) -> Path:
    """Run explicitly configured, resume-aware field generation without training."""

    config_path = _repo_path(config_path)
    cfg = _load_yaml(config_path)
    experiment = dict(cfg.get("experiment") or {})
    experiment_id = str(experiment["experiment_id"])
    output_cfg = dict(cfg.get("output") or {})
    resolved_run_dir = _repo_path(run_dir_override) if run_dir_override is not None else None
    run_dir = make_run_dir(
        experiment_id,
        runs_dir=_repo_path(output_cfg.get("runs_dir", "runs")),
        run_dir=resolved_run_dir,
    )
    logger = setup_logging(log_file=run_dir / "run.log", run_meta={"script": "scripts/run_field_eigen_te_generation.py", "config_path": str(config_path)})
    ensure_figures_dir(run_dir)
    started = time.perf_counter()
    history = [{"stage": "contract_check", "status": "started", "reason": ""}]
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    contract_path = _repo_path(str(cfg["contract_path"]))
    contract = _load_yaml(contract_path)
    if str(contract.get("contract", {}).get("id")) != "field_eigen_te_contract":
        raise ValueError("B1 generator requires the frozen field_eigen_te_contract.")
    allowed_contract_statuses = {"frozen_for_b1", "frozen_for_b1_with_b2_gated_scaleup", "frozen_for_b3_operator_data_and_adapted_pino", "b3_supervised_and_adapted_pino_gates_passed", "b4_formal_completed"}
    if str(contract.get("contract", {}).get("status")) not in allowed_contract_statuses:
        raise ValueError("Field contract does not permit B1 field generation.")

    source_cfg = dict(cfg.get("source_dataset") or {})
    source_path = _repo_path(str(source_cfg["path"]))
    with np.load(source_path, allow_pickle=False) as bundle:
        rho_all = np.asarray(bundle["rho"], dtype=np.float32)
        metadata_rows = _load_metadata(str(bundle["metadata_jsonl"].item()), int(rho_all.shape[0]))
        source_split_by_index = _source_split_map(bundle, int(rho_all.shape[0]))
        source_split_indices = {name: np.asarray(bundle[field_name], dtype=np.int64) for name, field_name in (("train", "idx_train"), ("val", "idx_val"), ("test", "idx_test"))}
    expected_hash = str(contract["task"]["structure_source"]["dataset_sha256"])
    if _file_hash(source_path) != expected_hash:
        raise ValueError("Source dataset SHA256 does not match the frozen field contract.")
    k_spec = load_dataset_k_path_spec(source_path)
    if k_spec is None:
        raise ValueError("B1 requires a dataset k-path sidecar.")
    k_validation_cfg = dict(cfg.get("k_path_validation") or {})
    if k_spec.metadata.verification_status in VERIFIED_K_PATH_STATUSES:
        k_validation = {
            "coordinate_verification": "source_sidecar_already_verified",
            "source_k_path_id": k_spec.k_path_id,
            "source_k_points_hash": k_spec.k_points_hash,
            "source_k_path_status": k_spec.metadata.verification_status,
        }
    else:
        reference_path_value = k_validation_cfg.get("verified_reference_dataset_path")
        if not reference_path_value:
            raise ValueError("B1 source k-path is not verified and no verified_reference_dataset_path was configured.")
        reference_path = _repo_path(str(reference_path_value))
        reference_spec = load_dataset_k_path_spec(reference_path)
        if reference_spec is None:
            raise ValueError("Configured verified k-path reference has no discoverable sidecar.")
        k_validation = validate_verified_k_path_reference(k_spec, reference_spec)
        k_validation["reference_dataset_path"] = str(reference_path.relative_to(ROOT))
    if k_spec.k_path_id != str(contract["task"]["k_path"]["k_path_id"]):
        raise ValueError("Dataset k-path id does not match the frozen field contract.")

    solver_cfg = {**dict(contract.get("solver") or {}), **dict(cfg.get("solver_override") or {})}
    solver = MpbBandSolverConfig(
        num_bands=int(solver_cfg["num_bands"]) if "num_bands" in solver_cfg else int(contract["task"]["band_selection"]["range_one_based"][1]),
        resolution=int(solver_cfg["resolution"]),
        polarization=str(solver_cfg.get("polarization", contract["task"]["polarization"])),
        epsilon_air=float(solver_cfg["epsilon_air"]),
        epsilon_material=float(solver_cfg["epsilon_material"]),
        lattice_constant=float(solver_cfg["lattice_constant"]),
        eigensolver_tolerance=float(solver_cfg["eigensolver_tolerance"]),
        band_guard_count=int(solver_cfg.get("band_guard_count", 0)),
        band_order_policy=str(solver_cfg["band_order_policy"]),
    )
    selection_manifest_path_value = cfg.get("selection_manifest_path")
    selection_manifest: dict[str, Any] | None = None
    if selection_manifest_path_value:
        selection_manifest_path = _repo_path(str(selection_manifest_path_value))
        selection_manifest = _load_selection_manifest(selection_manifest_path)
        if str(selection_manifest.get("dataset_sha256")) != expected_hash:
            raise ValueError("Field selection manifest does not bind the frozen source dataset hash.")
        if str(selection_manifest.get("k_points_hash")) != str(k_spec.k_points_hash):
            raise ValueError("Field selection manifest does not bind the active k-path coordinates.")
        if str(selection_manifest.get("contract_id")) != str(contract["contract"]["id"]):
            raise ValueError("Field selection manifest does not bind this field contract.")
        requested_records = [dict(row) for row in list(selection_manifest["records"])]
    elif cfg.get("selection"):
        selection_cfg = dict(cfg["selection"])
        common_selection = {
            "rho": rho_all,
            "metadata_rows": metadata_rows,
            "split_indices": source_split_indices,
            "k_spec": k_spec,
            "dataset_path": str(source_path.relative_to(ROOT)),
            "dataset_sha256": expected_hash,
            "contract_id": str(contract["contract"]["id"]),
            "contract_version": int(contract["contract"]["version"]),
            "n_occupancy_strata": int(selection_cfg["occupancy_strata"]),
            "n_k_strata": int(selection_cfg["k_strata"]),
            "band_range_one_based": contract["task"]["band_selection"]["range_one_based"],
            "selector_seed": int(selection_cfg["selector_seed"]),
        }
        if bool(selection_cfg.get("operator_learning", False)):
            selection_manifest = build_field_eigen_operator_selection_manifest(
                **common_selection,
                structure_split_counts=dict(selection_cfg["source_structure_split_counts"]),
                k_per_structure=int(selection_cfg["k_per_structure"]),
            )
        else:
            selection_manifest = build_field_eigen_selection_manifest(
                **common_selection,
                split_counts=dict(selection_cfg["source_split_counts"]),
            )
        selection_manifest_path_value = str(run_dir / "field_selection_manifest.json")
        write_json(Path(selection_manifest_path_value), selection_manifest)
        requested_records = [dict(row) for row in list(selection_manifest["records"])]
    else:
        requested_records = [dict(row) for row in list(cfg.get("records") or [])]
    if not requested_records:
        raise ValueError("B1 field generation requires an explicit non-empty records list.")
    pilot_target_count = int(contract["b1_handoff"]["initial_real_sample_count"])
    is_reference_audit = bool(cfg.get("reference_audit", False))
    is_scaleup = len(requested_records) > pilot_target_count and not is_reference_audit
    b2_gate_provenance: dict[str, Any] | None = None
    if is_scaleup:
        gate_cfg = dict(cfg.get("promotion_gate") or {})
        required_status = str(gate_cfg.get("required_status") or contract["b1_handoff"]["scaleup"]["requires_b2_residual_status"])
        if b2_gate_run_dir is None:
            raise ValueError("B1 scale-up requires --b2-gate-run-dir from a passing B2 residual run.")
        gate_dir = _repo_path(b2_gate_run_dir)
        gate_results_path = gate_dir / "results.json"
        if not gate_results_path.is_file():
            raise ValueError(f"B2 gate results were not found: {gate_results_path}")
        gate_results = json.loads(gate_results_path.read_text(encoding="utf-8"))
        if str(gate_results.get("status")) != required_status:
            raise ValueError(f"B1 scale-up requires B2 status {required_status}, got {gate_results.get('status')!r}.")
        b2_gate_provenance = {
            "run_dir": str(gate_dir.relative_to(ROOT)) if gate_dir.is_relative_to(ROOT) else str(gate_dir),
            "results_file_sha256": _file_hash(gate_results_path),
            "status": required_status,
        }
    resolved = {"generation": cfg, "contract": contract}
    write_resolved_config(run_dir, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    history[0] = {"stage": "contract_check", "status": "passed", "reason": ""}

    jobs: list[MPBWorkerJob] = []
    prepared: list[dict[str, Any]] = []
    range_start, range_end = (int(value) for value in contract["task"]["band_selection"]["range_one_based"])
    operator_manifest = bool(selection_manifest and str(selection_manifest.get("schema_version", "")).endswith(".v2"))
    grouped_requests: list[list[dict[str, Any]]]
    if operator_manifest:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for request in requested_records:
            grouped.setdefault(str(request["solver_job_id"]), []).append(request)
        grouped_requests = [sorted(value, key=lambda row: int(row["band_index_one_based"])) for _, value in sorted(grouped.items())]
    else:
        grouped_requests = [[request] for request in requested_records]

    for position, requests in enumerate(grouped_requests):
        request = requests[0]
        sample_index = int(request["source_sample_index"])
        k_index = int(request["k_index"])
        band_indices = [int(row["band_index_one_based"]) for row in requests]
        if not 0 <= sample_index < int(rho_all.shape[0]):
            raise ValueError(f"source_sample_index={sample_index} is outside source dataset")
        if not 0 <= k_index < int(k_spec.Nk):
            raise ValueError(f"k_index={k_index} is outside verified k-path")
        if any(not range_start <= band_index <= range_end for band_index in band_indices):
            raise ValueError("Requested band is outside the frozen B1 range")
        sample_meta = metadata_rows[sample_index]
        rho = rho_all[sample_index]
        k_point = np.asarray(k_spec.k_points[k_index], dtype=np.float64)
        source_split = str(request.get("source_split", source_split_by_index[sample_index]))
        if source_split != source_split_by_index[sample_index]:
            raise ValueError(f"Field selection source_split disagrees with source dataset for index {sample_index}.")
        job_id = str(request.get("solver_job_id") or f"s{sample_index:06d}_k{k_index:04d}_b{band_indices[0]:02d}")
        prepared.append(
            {
                "record_position": position,
                "solver_job_id": job_id,
                "source_sample_index": sample_index,
                "source_sample_id": str(sample_meta.get("sample_id", f"source_{sample_index:06d}")),
                "source_sample_hash": str(sample_meta.get("rho_hash") or hash_array(rho)),
                "source_split": source_split,
                "source_geometry_hash": sample_meta.get("geometry_hash"),
                "k_index": k_index,
                "k_point": k_point,
                "band_indices_one_based": band_indices,
                "rho": rho,
            }
        )
        jobs.append(
            MPBWorkerJob(
                sample_id=f"{experiment_id}_{job_id}",
                sample_index=position,
                split=source_split,
                rho=rho,
                k_points=[k_point.tolist()],
                solver_config=replace(
                    solver,
                    field_band_index_one_based=band_indices[0] if not operator_manifest else None,
                    field_band_indices_one_based=tuple(band_indices) if operator_manifest else None,
                ),
                rho_hash=str(sample_meta.get("rho_hash") or hash_array(rho)),
                geometry_hash=str(sample_meta.get("geometry_hash") or ""),
            )
        )

    parallel_cfg = MPBParallelConfig.from_mapping(dict(cfg.get("solver_workers") or {"workers": 1, "worker_backend": "subprocess"}))
    worker_results, worker_summary = run_mpb_worker_jobs(
        jobs,
        config=parallel_cfg,
        shard_dir=run_dir / "worker_shards",
        solver_fn=solve_eigenfield_multiband_bundle_for_rho_mpb if operator_manifest else solve_eigenfield_bundle_for_rho_mpb,
    )
    generated: list[FieldEigenRecord] = []
    by_position = {int(item["record_position"]): item for item in prepared}
    for result in worker_results:
        request = by_position[int(result.sample_index)]
        if result.status != "computed" or result.bands is None:
            failures.append({**{key: value for key, value in request.items() if key not in {"rho", "k_point"}}, "reason": result.error, "status": result.status})
            continue
        fields = np.asarray(result.bands)
        if not operator_manifest:
            fields = fields[None, ...]
        if fields.ndim < 3 or fields.shape[0] != len(request["band_indices_one_based"]) or fields.shape[1] != 2:
            failures.append({**{key: value for key, value in request.items() if key not in {"rho", "k_point"}}, "reason": f"unexpected worker field bundle shape {fields.shape}", "status": "invalid_worker_payload"})
            continue
        frequencies = list(result.solve_metadata.get("eigenfrequencies") or [result.solve_metadata.get("eigenfrequency")])
        solver_indices = list(result.solve_metadata.get("solver_band_indices_one_based") or [result.solve_metadata.get("solver_band_index_one_based")])
        epsilon = epsilon_from_rho(
            np.asarray(request["rho"]),
            eps_air=solver.epsilon_air,
            eps_alum=solver.epsilon_material,
        ).astype(np.float32)
        for band_position, band_index in enumerate(request["band_indices_one_based"]):
            efield, hfield, normalization = canonicalize_eigenfields(
                fields[band_position, 0],
                fields[band_position, 1],
                epsilon=epsilon,
                domain_extent=(solver.lattice_constant, solver.lattice_constant),
            )
            efield = efield.astype(np.complex64); hfield = hfield.astype(np.complex64)
            record_id = f"{experiment_id}_s{int(request['source_sample_index']):06d}_k{int(request['k_index']):04d}_b{int(band_index):02d}"
            provenance = {
                **dict(result.solve_metadata),
                "source_dataset_path": str(source_path.relative_to(ROOT)), "source_dataset_sha256": expected_hash,
                "source_sample_index": request["source_sample_index"], "source_sample_id": request["source_sample_id"], "source_split": request["source_split"],
                "source_geometry_hash": request["source_geometry_hash"], "geometry_hash": request["source_geometry_hash"],
                "k_path_id": k_spec.k_path_id, "k_points_hash": k_spec.k_points_hash, "k_path_hash": hash_k_path_spec(k_spec), "k_path_validation": k_validation,
                "mpb_solve_time_s": float(result.solve_time_s), "solver_job_id": request["solver_job_id"],
                "requested_band_index_one_based": int(band_index), "solver_band_index_one_based": int(solver_indices[band_position]),
            }
            record = FieldEigenRecord(
                record_id=record_id, source_sample_id=str(request["source_sample_id"]), source_sample_hash=str(request["source_sample_hash"]), source_split=str(request["source_split"]),
                rho=np.asarray(request["rho"], dtype=np.float32), epsilon=epsilon,
                k_point=np.asarray(request["k_point"], dtype=np.float64), k_path_id=k_spec.k_path_id, band_index_one_based=int(band_index), eigenfrequency=float(frequencies[band_position]),
                efield_complex=efield, hfield_complex=hfield, polarization=solver.polarization.upper(), bloch_phase=np.asarray(request["k_point"], dtype=np.float64), lattice_constant=solver.lattice_constant,
                domain_extent=np.asarray([solver.lattice_constant, solver.lattice_constant], dtype=np.float64), dx=float(normalization["dx"]), dy=float(normalization["dy"]), field_normalization=normalization, solver_provenance=provenance,
            )
            generated.append(record)
            allocated_solve_time = float(result.solve_time_s) / max(len(request["band_indices_one_based"]), 1)
            rows.append({"record_id": record.record_id, "solver_job_id": request["solver_job_id"], "source_sample_index": request["source_sample_index"], "source_sample_id": record.source_sample_id, "source_split": record.source_split, "k_index": request["k_index"], "band_index_one_based": record.band_index_one_based, "eigenfrequency": record.eigenfrequency, "mpb_solve_time_s": allocated_solve_time, "efield_hash": hash_array(efield), "hfield_hash": hash_array(hfield), "status": "generated", "resumed": bool(result.resumed)})
            logger.info("Generated field record %s", record.record_id)

    _write_jsonl(run_dir / "failed_case_ledger.jsonl", failures)
    _write_jsonl(run_dir / "sample_metadata.jsonl", rows)
    write_json(
        run_dir / "solver_provenance.json",
        {"contract_id": contract["contract"]["id"], "environment": solver_cfg["environment"], "records": [record.solver_provenance for record in generated]},
    )

    dataset_path = run_dir / str(output_cfg.get("field_filename", "field_eigen_te_preflight.h5"))
    is_b1_pilot = selection_manifest is not None and len(requested_records) == pilot_target_count
    status = (
        "b4_resolution_reference_data_ready"
        if not failures and len(generated) == len(requested_records) and is_reference_audit
        else
        "b1_64_field_data_pilot_pass"
        if not failures and len(generated) == len(requested_records) and is_b1_pilot
        else "b3_9216_field_data_ready"
        if not failures and len(generated) == len(requested_records) and operator_manifest
        else "b1_1000_field_data_ready"
        if not failures and len(generated) == len(requested_records) and is_scaleup
        else "real_mpb_preflight_pass"
        if not failures and len(generated) == len(requested_records)
        else "field_generation_incomplete"
    )
    field_audit: dict[str, Any] = {"record_count": len(generated), "normalization_failures": [], "provenance_failures": []}
    required_provenance = set(contract["record_schema"]["solver_provenance_required"])
    for record in generated:
        epsilon_field = np.repeat(
            np.repeat(record.epsilon, record.efield_complex.shape[0] // record.epsilon.shape[0], axis=0),
            record.efield_complex.shape[1] // record.epsilon.shape[1],
            axis=1,
        )
        while epsilon_field.ndim < record.efield_complex.ndim:
            epsilon_field = epsilon_field[..., None]
        field_l2 = float(np.sqrt(np.sum(epsilon_field * np.abs(record.efield_complex) ** 2) * record.dx * record.dy))
        # Use the deterministic anchor selected during canonicalization. At
        # symmetry points several components can have equal magnitude; after
        # complex64 storage, recomputing argmax may select a different tied
        # component even though the canonical anchor remains valid.
        anchor_index = int(record.field_normalization["anchor_flat_index"])
        anchor = record.efield_complex.reshape(-1)[anchor_index]
        if not np.isclose(field_l2, 1.0, rtol=1.0e-5, atol=1.0e-6) or abs(float(np.imag(anchor))) > 1.0e-5 or float(np.real(anchor)) < 0.0:
            field_audit["normalization_failures"].append(record.record_id)
        missing_provenance = sorted(required_provenance - set(record.solver_provenance))
        if missing_provenance:
            field_audit["provenance_failures"].append({"record_id": record.record_id, "missing": missing_provenance})
    if field_audit["normalization_failures"] or field_audit["provenance_failures"]:
        raise RuntimeError("Generated fields failed the frozen normalization or provenance audit.")
    if not failures and len(generated) == len(requested_records):
        write_field_eigen_hdf5(
            dataset_path,
            generated,
            metadata={
                "contract_id": contract["contract"]["id"],
                "status": status,
                "boundary_condition": "bloch_periodic",
                "source_dataset_path": str(source_path.relative_to(ROOT)),
                "source_dataset_sha256": expected_hash,
                "k_path_id": k_spec.k_path_id,
                "k_path_validation": k_validation,
                "selection_manifest_path": str(selection_manifest_path_value or ""),
                "selection_manifest_sha256": selection_manifest.get("manifest_sha256") if selection_manifest else None,
                "b2_gate_provenance": b2_gate_provenance,
            },
        )
        schema = inspect_field_dataset(dataset_path)
        required = set(contract["record_schema"]["required_fields"])
        present = set(schema.dataset_names) | {"boundary_condition", "field_normalization", "solver_provenance", "domain_extent", "dx", "dy"}
        missing = sorted({"eigenfrequency", "efield_complex", "hfield_complex", "bloch_phase", "source_split"} - present)
        if missing:
            raise RuntimeError(f"Written field dataset is missing required schema fields: {missing}")
    else:
        schema = None

    manifest = {
        "dataset_path": str(dataset_path.relative_to(ROOT)) if dataset_path.exists() else "",
        "dataset_file_hash": _file_hash(dataset_path) if dataset_path.exists() else None,
        "status": status,
        "record_count": len(generated),
        "requested_record_count": len(requested_records),
        "contract_id": contract["contract"]["id"],
        "source_dataset": str(source_path.relative_to(ROOT)),
        "k_path_id": k_spec.k_path_id,
        "selection_manifest_path": str(selection_manifest_path_value or ""),
        "selection_manifest_sha256": selection_manifest.get("manifest_sha256") if selection_manifest else None,
        "solver_provenance": "solver_provenance.json",
        "sample_metadata": "sample_metadata.jsonl",
        "worker_summary": "worker_summary.json",
        "failed_case_ledger": "failed_case_ledger.jsonl",
        "schema_dataset_names": list(schema.dataset_names) if schema is not None else [],
        "b2_gate_provenance": b2_gate_provenance,
        "next_gate": "B2_residual_unit_tests_and_solver_reference_vs_random_field" if is_b1_pilot else "B3_supervised_v2_and_pino_three_seed" if operator_manifest else "B2_supervised_field_baseline_three_seed",
    }
    _write_yaml(run_dir / "dataset_manifest.yaml", manifest)
    history.append({"stage": "b1_field_generation", "status": status, "reason": failures[0]["reason"] if failures else ""})
    _write_history(run_dir / "history.csv", history)
    elapsed = float(time.perf_counter() - started)
    metrics = {
        "status": status,
        "record_count": len(generated),
        "requested_record_count": len(requested_records),
        "wall_time_total_s": elapsed,
        "mpb_generation_time_s": float(sum(float(row["mpb_solve_time_s"]) for row in rows)),
        "mpb_generation_time_per_record_s": float(sum(float(row["mpb_solve_time_s"]) for row in rows) / max(len(rows), 1)),
        "real_mpb_generation_started": True,
        "real_mpb_generation_completed": status in {"real_mpb_preflight_pass", "b1_64_field_data_pilot_pass", "b1_1000_field_data_ready", "b3_9216_field_data_ready", "b4_resolution_reference_data_ready"},
        "field_data_pilot_pass": status == "b1_64_field_data_pilot_pass",
        "field_data_training_ready": status in {"b1_1000_field_data_ready", "b3_9216_field_data_ready"},
        "solver_job_count": len(grouped_requests),
        "field_model_training_started": False,
        "records_per_source_split": dict(Counter(record.source_split for record in generated)),
        "records_per_band": {str(key): int(value) for key, value in sorted(Counter(record.band_index_one_based for record in generated).items())},
        "field_contract_audit": field_audit,
        **worker_summary,
    }
    write_metrics(run_dir, metrics)
    write_results(run_dir, {"action": "field_eigen_te_generation", "status": status, "dataset_manifest": "dataset_manifest.yaml", "dataset_written": dataset_path.exists(), "next_gate": manifest["next_gate"]})
    write_json(run_dir / "field_contract_audit.json", field_audit)
    write_run_meta(run_dir, {"script": "scripts/run_field_eigen_te_generation.py", "config_path": str(config_path.relative_to(ROOT)), "validation_boundary": "wsl_meep_env_real_mpb_eigenfield_generation", "status": status})
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Generate contract-complete TE Bloch eigenfield records with real MPB.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--b2-gate-run-dir", type=Path, default=None, help="Passing B2 residual run required for scale-up datasets.")
    args = parser.parse_args(argv)
    run_dir = run_generation(args.config, run_dir_override=args.run_dir, b2_gate_run_dir=args.b2_gate_run_dir)
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
