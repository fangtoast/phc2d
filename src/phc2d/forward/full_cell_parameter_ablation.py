"""Freeze and audit the Full-cell Full-Wh parameter-ablation campaign.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from phc2d.runs import (
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)


EXPECTED_MODELS = {
    "mlp",
    "res_mlp_wide",
    "lifted_cnn",
    "fno2d_band_v2",
    "mlp_mixer_band_v1",
    "res_mlp_gated_v2",
    "convnext_band_v1",
    "tfno2d_band_v3",
    "dual_lane_band_operator_v1",
}
LEDGER_FIELDS = (
    "model",
    "lane",
    "parameter",
    "category",
    "current_value_json",
    "source_config",
    "source_code",
    "evidence_grade",
    "impacts_json",
    "minimum_search_values_json",
    "fixed_confounders_json",
    "validation_metric",
    "stop_rule",
)
MANIFEST_FIELDS = (
    "stage",
    "case_id",
    "lane",
    "model",
    "parameter",
    "value_json",
    "is_current",
    "seed",
    "max_epochs",
    "admission_probe",
    "derived_config",
    "run_dir",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _dump_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _slug(value: Any) -> str:
    raw = _json(value).lower()
    return "".join(char if char.isalnum() else "-" for char in raw).strip("-")


def _source_config_path(contract: Mapping[str, Any], lane: str, group: str) -> str:
    dataset = contract["datasets"][lane]
    if group == "upgrade":
        return str(dataset["upgrade_config"])
    if group == "baseline":
        return str(dataset["source_config"])
    if group == "dual":
        return str(contract["datasets"]["joint"]["source_config"])
    raise ValueError(f"unsupported config_group: {group}")


def _model_params(source: Mapping[str, Any], model: str, group: str) -> dict[str, Any]:
    if group == "dual":
        if source.get("model", {}).get("name") != model:
            raise ValueError(f"dual source config does not define {model}")
        return dict(source["model"].get("params") or {})
    models = source.get("models") or {}
    if model not in models:
        raise ValueError(f"source config does not define {model}")
    return dict(models[model])


def _validate_contract(contract: Mapping[str, Any], root: Path) -> None:
    if contract.get("schema_version") != "phc2d.full_cell_parameter_ablation.v1":
        raise ValueError("unsupported parameter-ablation schema_version")
    models = set((contract.get("models") or {}).keys())
    if models != EXPECTED_MODELS:
        raise ValueError(f"model inventory mismatch: {sorted(models ^ EXPECTED_MODELS)}")
    representation = contract.get("representation") or {}
    if representation.get("input_field") != "material_weight_grid_64":
        raise ValueError("input field must be material_weight_grid_64")
    if representation.get("target_field") != "Wh" or representation.get("target_shape") != [81, 10]:
        raise ValueError("target contract must be Wh[81,10]")
    split = (contract.get("split") or {}).get("per_lane") or {}
    if split != {"train": 40000, "validation": 5000, "held_out_test": 5000}:
        raise ValueError("split must be exactly 40k/5k/5k per lane")
    for lane in ("binary", "graded"):
        dataset = contract["datasets"][lane]
        if Path(str(dataset["path"])).name != f"full_cell_{lane}_bz9_50000_xingyi_v1.h5":
            raise ValueError(f"unexpected {lane} dataset path")
    for model, model_cfg in contract["models"].items():
        group = str(model_cfg["config_group"])
        for lane in model_cfg["lanes"]:
            source_path = root / _source_config_path(contract, lane, group)
            source = _load_yaml(source_path)
            params = _model_params(source, model, group)
            wave1 = 0
            for parameter, ablation in model_cfg["critical_parameters"].items():
                if ablation.get("wave1"):
                    wave1 += 1
                current = ablation["current"]
                if parameter == "learning_rate":
                    actual = (
                        source["training"].get("learning_rate")
                        if group == "dual"
                        else source["training"]["learning_rates"][model]
                    )
                else:
                    actual = params.get(parameter)
                if actual != current:
                    raise ValueError(
                        f"current-value drift for {model}.{parameter}: contract={current!r}, source={actual!r}"
                    )
                if current not in ablation["values"]:
                    raise ValueError(f"search values omit current value for {model}.{parameter}")
            if wave1 != 1:
                raise ValueError(f"{model} must designate exactly one wave1 parameter")


def _derived_config(
    source: Mapping[str, Any],
    *,
    model: str,
    group: str,
    lane: str,
    stage: str,
    parameter: str | None,
    value: Any,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    derived = deepcopy(dict(source))
    experiment = dict(derived.get("experiment") or {})
    experiment.update(
        {
            "experiment_id": f"full_cell_parameter_ablation_{stage}_{lane}_{model}",
            "family": "full_cell_forward_parameter_ablation",
            "evidence_status": "single_batch_health_probe" if stage == "a" else "single_seed_short_budget_screen",
            "parent_experiment_id": source.get("experiment", {}).get("experiment_id"),
            "parameter_under_test": parameter,
            "parameter_value": value,
            "held_out_test_authorized": False,
        }
    )
    derived["experiment"] = experiment
    training = dict(derived.get("training") or {})
    training["seed"] = int(protocol["seed"])
    training["require_finite_gradients"] = True
    if stage == "b":
        stage_b = protocol["stage_b"]
        training.update(
            {
                "train_size": int(stage_b["train_size"]),
                "max_epochs": int(stage_b["max_epochs"]),
                "early_stopping_patience": int(stage_b["early_stopping_patience"]),
                "epoch_100_continue_gate": False,
            }
        )
    derived["training"] = training
    if parameter is not None:
        if parameter == "learning_rate":
            if group == "dual":
                training["learning_rate"] = value
            else:
                learning_rates = dict(training.get("learning_rates") or {})
                learning_rates[model] = value
                training["learning_rates"] = learning_rates
        elif group == "dual":
            params = dict(derived["model"].get("params") or {})
            params[parameter] = value
            derived["model"]["params"] = params
        else:
            models = dict(derived.get("models") or {})
            params = dict(models[model])
            params[parameter] = value
            models[model] = params
            derived["models"] = models
    return derived


def freeze_parameter_ablation(config_path: str | Path, output_dir: str | Path, *, root: str | Path = ".") -> Path:
    """Freeze ledgers, derived configs, and stage manifests without training."""

    config_file = Path(config_path).resolve()
    repository_root = Path(root).resolve()
    contract = _load_yaml(config_file)
    _validate_contract(contract, repository_root)
    out = make_run_dir("full_cell_forward_parameter_ablation_v1", run_dir=output_dir)
    ensure_figures_dir(out)
    setup_logging(log_file=out / "run.log")
    ledger: list[dict[str, Any]] = []
    full_matrix: list[dict[str, Any]] = []
    stage_a: list[dict[str, Any]] = []
    stage_b: list[dict[str, Any]] = []
    protocol = contract["protocol"]
    fixed = protocol["fixed"]
    configs_dir = out / "derived_configs"
    for model, model_cfg in contract["models"].items():
        group = str(model_cfg["config_group"])
        for lane in model_cfg["lanes"]:
            source_rel = _source_config_path(contract, lane, group)
            source = _load_yaml(repository_root / source_rel)
            for parameter, ablation in model_cfg["critical_parameters"].items():
                grade = "pre-registered fair setting" if parameter == "learning_rate" else "unvalidated"
                ledger.append(
                    {
                        "model": model,
                        "lane": lane,
                        "parameter": parameter,
                        "category": ablation["category"],
                        "current_value_json": _json(ablation["current"]),
                        "source_config": source_rel,
                        "source_code": contract["evidence_sources"]["dual_runner" if group == "dual" else "runner"],
                        "evidence_grade": grade,
                        "impacts_json": _json(ablation.get("impacts", [])),
                        "minimum_search_values_json": _json(ablation["values"]),
                        "fixed_confounders_json": _json(fixed),
                        "validation_metric": protocol["stage_b"]["selection_metric"],
                        "stop_rule": "20 epochs; fail on non-finite loss/gradient; no held-out test",
                    }
                )
                for value in ablation["values"]:
                    case_id = f"b-{lane}-{model}-{parameter}-{_slug(value)}"
                    derived_path = configs_dir / f"{case_id}.yaml"
                    _dump_yaml(
                        derived_path,
                        _derived_config(
                            source,
                            model=model,
                            group=group,
                            lane=lane,
                            stage="b",
                            parameter=parameter,
                            value=value,
                            protocol=protocol,
                        ),
                    )
                    row = {
                        "stage": "b",
                        "case_id": case_id,
                        "lane": lane,
                        "model": model,
                        "parameter": parameter,
                        "value_json": _json(value),
                        "is_current": str(value == ablation["current"]).lower(),
                        "seed": protocol["seed"],
                        "max_epochs": protocol["stage_b"]["max_epochs"],
                        "admission_probe": "false",
                        "derived_config": str(derived_path.resolve()),
                        "run_dir": str((out / "stage_b" / case_id).resolve()),
                    }
                    full_matrix.append(row)
                    if ablation.get("wave1"):
                        stage_b.append(row)
            case_id = f"a-{lane}-{model}-current"
            derived_path = configs_dir / f"{case_id}.yaml"
            _dump_yaml(
                derived_path,
                _derived_config(
                    source,
                    model=model,
                    group=group,
                    lane=lane,
                    stage="a",
                    parameter=None,
                    value=None,
                    protocol=protocol,
                ),
            )
            stage_a.append(
                {
                    "stage": "a",
                    "case_id": case_id,
                    "lane": lane,
                    "model": model,
                    "parameter": "current_contract",
                    "value_json": "null",
                    "is_current": "true",
                    "seed": protocol["seed"],
                    "max_epochs": 1,
                    "admission_probe": "true",
                    "derived_config": str(derived_path.resolve()),
                    "run_dir": str((out / "stage_a" / case_id).resolve()),
                }
            )
    _write_csv(out / "parameter_ledger.csv", LEDGER_FIELDS, ledger)
    _write_csv(out / "ablation_matrix.csv", MANIFEST_FIELDS, full_matrix)
    _write_csv(out / "stage_a_manifest.csv", MANIFEST_FIELDS, stage_a)
    _write_csv(out / "stage_b_wave1_manifest.csv", MANIFEST_FIELDS, stage_b)
    write_resolved_config(out, yaml.safe_dump(contract, sort_keys=False))
    (out / "history.csv").write_text("stage,status\nfreeze,complete\n", encoding="utf-8")
    write_metrics(
        out,
        {
            "ledger_rows": len(ledger),
            "full_matrix_cases": len(full_matrix),
            "stage_a_cases": len(stage_a),
            "stage_b_wave1_cases": len(stage_b),
            "gpu_concurrency": protocol["stage_a"]["gpu_concurrency"],
        },
    )
    write_results(
        out,
        {
            "status": "parameter_ablation_frozen",
            "test_executed": False,
            "stage_a_authorized": True,
            "stage_b_authorized_after_stage_a_gate": True,
            "stage_c_authorized": False,
        },
    )
    write_run_meta(
        out,
        {
            "script": "python -m phc2d.forward.full_cell_parameter_ablation freeze",
            "config_path": str(config_file),
            "primary_layer": "L3",
            "test_executed": False,
        },
    )
    return out


def audit_stage(manifest_path: str | Path, output_path: str | Path) -> Path:
    """Fail closed unless every manifest run is complete and test-free."""

    manifest_file = Path(manifest_path)
    rows = list(csv.DictReader(manifest_file.open(encoding="utf-8", newline="")))
    if not rows:
        raise ValueError("manifest is empty")
    failures: list[dict[str, str]] = []
    audited: list[dict[str, Any]] = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        result_path = run_dir / "results.json"
        metrics_path = run_dir / "metrics.json"
        if not result_path.is_file() or not metrics_path.is_file():
            failures.append({"case_id": row["case_id"], "reason": "missing_results_or_metrics"})
            continue
        results = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if bool(results.get("test_executed", False)):
            failures.append({"case_id": row["case_id"], "reason": "held_out_test_accessed"})
            continue
        status = str(results.get("status", ""))
        if not status.endswith("success") and not status.endswith("probe_pass"):
            failures.append({"case_id": row["case_id"], "reason": f"terminal_status={status or 'missing'}"})
            continue
        audited.append({**row, "terminal_status": status, "metrics": metrics})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "status": "pass" if not failures and len(audited) == len(rows) else "fail",
                "expected_cases": len(rows),
                "audited_cases": len(audited),
                "failures": failures,
                "test_executed": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output


def run_manifest_case(manifest_path: str | Path, index: int) -> Path:
    """Run one frozen manifest row selected by its zero-based index."""

    manifest_file = Path(manifest_path)
    rows = list(csv.DictReader(manifest_file.open(encoding="utf-8", newline="")))
    if index < 0 or index >= len(rows):
        raise IndexError(f"manifest index {index} outside 0..{len(rows) - 1}")
    row = rows[index]
    config = Path(row["derived_config"])
    run_dir = Path(row["run_dir"])
    admission_probe = row["admission_probe"].lower() == "true"
    if row["lane"] == "joint":
        from phc2d.forward.full_cell_dual_lane import run_full_cell_dual_lane

        return run_full_cell_dual_lane(
            config,
            seed=int(row["seed"]),
            head_kind="dense",
            run_dir=run_dir,
            admission_probe=admission_probe,
        )
    from phc2d.forward.full_cell import run_full_cell_forward

    return run_full_cell_forward(
        config,
        model_name=row["model"],
        seed=int(row["seed"]),
        run_dir=run_dir,
        admission_probe=admission_probe,
        max_epochs=None if admission_probe else int(row["max_epochs"]),
        train_size=None if admission_probe else 40000,
        head_kind="dense",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--root", type=Path, default=Path("."))
    audit = subparsers.add_parser("audit-stage")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    run_case = subparsers.add_parser("run-case")
    run_case.add_argument("--manifest", type=Path, required=True)
    run_case.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(freeze_parameter_ablation(args.config, args.output_dir, root=args.root))
    elif args.command == "audit-stage":
        print(audit_stage(args.manifest, args.output))
    else:
        print(run_manifest_case(args.manifest, args.index))


if __name__ == "__main__":
    main()


__all__ = ["audit_stage", "freeze_parameter_ablation", "run_manifest_case"]
