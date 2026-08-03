"""Pilot dataset split audit for MVP-0.

Writes run artifacts under the standard phc2d run directory layout:
``runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`` with
`metrics.json`, `results.json`, `run.meta.json`, and `run.log`.
"""

from __future__ import annotations

from collections import Counter
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from phc2d.data import DataRepresentationSpec, infer_dataset_schema, load_forward_dataset, validate_representation_compatibility
from phc2d.runs import (
    make_run_dir,
    setup_logging,
    write_metrics,
    write_results,
    write_resolved_config,
    write_run_meta,
)


def _to_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid YAML payload: {path}")
    return payload


def _load_npz_payload(dataset_path: Path) -> dict[str, Any]:
    bundle = load_forward_dataset(dataset_path)
    sample_metadata = bundle.metadata
    sample_count = len(sample_metadata)
    split_distribution = dict(Counter(str(item.get("split", "unassigned")) for item in sample_metadata)) if sample_count else {}
    source_distribution = dict(
        Counter(str(item.get("sample_id_source", "missing")) for item in sample_metadata)
    ) if sample_count else {}

    return {
        "n_samples": bundle.n_samples,
        "rho_shape": [int(v) for v in bundle.rho.shape],
        "Wh_shape": [int(v) for v in bundle.Wh.shape],
        "split_counts": {
            "train": int(bundle.idx_train.size),
            "val": int(bundle.idx_val.size),
            "test": int(bundle.idx_test.size),
        },
        "metadata_entries": int(len(bundle.metadata)),
        "sample_id_available": (
            all(isinstance(item, dict) and "sample_id" in item for item in bundle.metadata)
            if bundle.metadata
            else False
        ),
        "sample_id_coverage": float(sum(1 for item in sample_metadata if isinstance(item.get("sample_id"), str) and item.get("sample_id")))
        / max(sample_count, 1),
        "split_distribution": split_distribution,
        "sample_id_source_distribution": source_distribution,
    }


def _audit_dataset(schema: dict[str, Any], *, has_gap_fields: bool) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for input_repr in ("rho_wedge", "rho_grid", "epsilon_grid"):
        for output_target in ("Wh", "band_query", "gap_descriptors"):
            spec = DataRepresentationSpec(input_repr=input_repr, output_target=output_target)
            problems = validate_representation_compatibility(schema, spec)
            checks[f"{input_repr}->{output_target}"] = len(problems) == 0
    checks["gap_fields_present"] = bool(has_gap_fields)
    return checks


def _build_dataset_record(path: Path) -> dict[str, Any]:
    schema = infer_dataset_schema(path)
    has_gap_fields = False
    fields = schema.get("fields", {})
    if isinstance(fields, dict):
        has_gap_fields = any(key.startswith("idx_gap_") for key in fields.keys())

    if path.suffix.lower() == ".npz":
        payload = _load_npz_payload(path)
    else:
        # HDF5 or field-style data uses schema-only checks.
        payload = {
            "n_samples": int(schema.get("n_samples", 0)),
            "rho_shape": schema.get("rho_shape"),
            "Wh_shape": None,
            "split_counts": {
                "train": 0,
                "val": 0,
                "test": 0,
            },
            "metadata_entries": 0,
            "sample_id_available": False,
            "sample_id_coverage": 0.0,
            "split_distribution": {},
            "sample_id_source_distribution": {},
        }

    payload.update(
        {
            "path": str(path),
            "kind": schema.get("kind"),
            "rho_shape_legacy": schema.get("rho_shape"),
            "wh_shape_legacy": schema.get("wh_shape"),
            "k_points": schema.get("k_points"),
            "n_bands": schema.get("n_bands"),
            "compatibility": _audit_dataset(schema, has_gap_fields=has_gap_fields),
            "fields": sorted(list(schema.get("fields", {}).keys())),
        }
    )
    return payload


def main() -> None:
    config_path = Path("configs/research/dataset_split_audit.yaml")
    config = _to_mapping(config_path)
    experiment_id = str(
        config.get("experiment", {})
        .get("experiment_id", "dataset_split_audit")
    )
    runs_dir = config.get("output", {}).get("runs_dir", "runs")
    run_dir = make_run_dir(experiment_id, runs_dir=str(runs_dir))
    setup_logging(log_file=run_dir / "run.log", run_meta={"script": "MVP-0 dataset split audit"})
    write_run_meta(run_dir, {"script": "MVP-0 dataset split audit", "dataset_config": config_path.as_posix()})
    write_resolved_config(run_dir, yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    datasets = [Path(item) for item in config.get("dataset", {}).get("paths", [])]
    dataset_records: list[dict[str, Any]] = []
    for dataset_path in datasets:
        dataset_records.append(_build_dataset_record(dataset_path))

    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(dataset_records),
        "datasets_ok": int(sum(1 for item in dataset_records if item["n_samples"] > 0)),
    }

    results = {
        "status": "ok" if dataset_records else "blocked",
        "experiment_id": experiment_id,
        "stage": "dataset_split_audit",
        "run_dir": str(run_dir),
        "datasets": dataset_records,
        "notes": [
            "Schema checks are performed via phc2d.data.infer_dataset_schema",
            "sample_id/lineage fields are inferred when missing in metadata_jsonl",
        ],
    }
    write_metrics(run_dir, metrics)
    write_results(run_dir, results)

    report_dir = Path("docs/results")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"dataset_split_audit_{datetime.now().strftime('%Y%m%d')}.md"
    lines = ["# Dataset split audit (MVP-0)", "", f"- run_dir: {run_dir}"]
    for item in dataset_records:
        lines.extend(
            [
                "",
                f"## {item['path']}",
                f"- n_samples: {item['n_samples']}",
                f"- splits: {item['split_counts']}",
                f"- rho_shape: {item['rho_shape']}",
                f"- Wh_shape: {item['Wh_shape']}",
                f"- sample_id_available: {item['sample_id_available']}",
                f"- sample_id_coverage: {item['sample_id_coverage']:.3f}",
                f"- split_distribution: {item['split_distribution']}",
                f"- sample_id_source_distribution: {item['sample_id_source_distribution']}",
                f"- gap_fields_present: {item['compatibility']['gap_fields_present']}",
                f"- compatibility rho_wedge->Wh: {item['compatibility']['rho_wedge->Wh']}",
                f"- compatibility rho_grid->Wh: {item['compatibility']['rho_grid->Wh']}",
                f"- compatibility epsilon_grid->Wh: {item['compatibility']['epsilon_grid->Wh']}",
                f"- compatibility rho_wedge->band_query: {item['compatibility']['rho_wedge->band_query']}",
                f"- compatibility rho_grid->band_query: {item['compatibility']['rho_grid->band_query']}",
                f"- compatibility epsilon_grid->band_query: {item['compatibility']['epsilon_grid->band_query']}",
                f"- compatibility rho_wedge->gap_descriptors: {item['compatibility']['rho_wedge->gap_descriptors']}",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
