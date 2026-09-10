"""Freeze EXP-FWD-019 seed0 component-ablation tasks from C0 evidence.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


FAMILIES = {
    "fno": ("fno2d_band_component_v1", "fno2d_band_v2"),
    "tfno": ("tfno2d_band_component_v1", "tfno2d_band_v3"),
    "convnext": ("convnext_band_component_v1", "convnext_band_v1"),
    "joint": ("dual_lane_band_component_v1", "dual_lane_band_operator_v1"),
}
SOURCES = {
    ("fno", "binary"): "configs/research/full_cell_forward_binary_native_50000_v1.yaml",
    ("fno", "graded"): "configs/research/full_cell_forward_graded_continuous_50000_v1.yaml",
    ("tfno", "binary"): "configs/research/full_cell_forward_upgrade_binary_50k_v1.yaml",
    ("tfno", "graded"): "configs/research/full_cell_forward_upgrade_graded_50k_v1.yaml",
    ("convnext", "binary"): "configs/research/full_cell_forward_upgrade_binary_50k_v1.yaml",
    ("convnext", "graded"): "configs/research/full_cell_forward_upgrade_graded_50k_v1.yaml",
    ("joint", "joint"): "configs/research/full_cell_forward_upgrade_dual_lane_50k_v1.yaml",
}
CANDIDATE_PARAMS = {
    "fno": {
        "no_primary_spectral": {"use_primary_spectral": False, "use_band_spectral": False},
        "no_band_spectral": {"use_band_spectral": False},
        "no_band_gate": {"use_band_gate": False},
        "no_low_rank": {"use_low_rank": False},
    },
    "tfno": {
        "no_spectral": {"enabled_branches": ["local", "channel"]},
        "no_local": {"enabled_branches": ["spectral", "channel"]},
        "no_channel": {"enabled_branches": ["spectral", "local"]},
    },
    "convnext": {
        "kernel_1": {"kernel_size": 1},
        "zero_padding": {"padding_mode": "zeros"},
        "no_hierarchical_downsample": {"hierarchical_downsample": False},
    },
    "joint": {
        "no_spectral": {"enabled_branches": ["local", "channel"]},
        "no_local": {"enabled_branches": ["spectral", "channel"]},
        "no_channel": {"enabled_branches": ["spectral", "local"]},
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _mask_score(metrics: Mapping[str, Any], mask: str) -> float:
    values = []
    for lane in metrics["lanes"].values():
        values.append(float(lane["perturbations"][f"mask_{mask}"]["prediction_delta_relative_pct"]))
    return float(sum(values) / len(values))


def _select_candidates(c0: Path) -> dict[str, list[str]]:
    cases = {name: _read_json(c0 / name / "metrics.json") for name in ("fno_b", "fno_g", "tfno_b", "tfno_g", "joint")}
    for name in ("fno_b", "fno_g", "tfno_b", "tfno_g", "convnext_b", "convnext_g", "joint"):
        result = _read_json(c0 / name / "results.json")
        if result.get("status") != "frozen_validation_diagnostic_success" or result.get("test_targets_loaded") is not False:
            raise RuntimeError(f"C0 case is not a successful validation-only diagnostic: {name}")
    selected: dict[str, list[str]] = {}
    mappings = {
        "fno": {"primary_spectral": "no_primary_spectral", "band_spectral": "no_band_spectral", "band_gate": "no_band_gate", "low_rank": "no_low_rank"},
        "tfno": {"spectral": "no_spectral", "local": "no_local", "channel": "no_channel"},
        "joint": {"spectral": "no_spectral", "local": "no_local", "channel": "no_channel"},
    }
    for family, mask_map in mappings.items():
        family_cases = [cases["joint"]] if family == "joint" else [cases[f"{family}_b"], cases[f"{family}_g"]]
        scores = {candidate: sum(_mask_score(metrics, mask) for metrics in family_cases) / len(family_cases) for mask, candidate in mask_map.items()}
        selected[family] = sorted(scores, key=lambda name: (-scores[name], name))[:2]
    selected["convnext"] = ["kernel_1", "zero_padding"]
    return selected


def _source_params(source: Mapping[str, Any], source_name: str) -> dict[str, Any]:
    if "models" in source:
        return dict(source["models"][source_name])
    return dict(source["model"]["params"])


def _derived_config(source: Mapping[str, Any], *, family: str, lane: str, variant: str) -> dict[str, Any]:
    config = deepcopy(dict(source))
    model_name, source_name = FAMILIES[family]
    params = _source_params(source, source_name)
    if family == "fno":
        params.update({"head_kind": "dense", "head_hidden_dim": 256, "band_embedding_dim": 32, "fourier_frequencies": [1, 2, 4, 8], "dropout": 0.05})
    params.update(CANDIDATE_PARAMS[family].get(variant, {}))
    if lane == "joint":
        config["model"] = {"name": model_name, "params": params}
        config["training"]["learning_rate"] = float(source["training"]["learning_rate"])
    else:
        source_lr = float(source["training"]["learning_rates"][source_name])
        config["models"] = {model_name: params}
        config["training"]["learning_rates"] = {model_name: source_lr}
    config["experiment"].update({
        "experiment_id": "EXP-FWD-019", "family": "full_cell_component_attribution",
        "evidence_status": "single_seed_short_budget_screen", "component_family": family,
        "component_variant": variant, "held_out_test_authorized": False,
    })
    config["training"].update({"max_epochs": 20, "early_stopping_patience": 0, "epoch_100_continue_gate": False})
    return config


def _confirmation_config(source: Mapping[str, Any], *, family: str, lane: str, variant: str, seed: int) -> dict[str, Any]:
    config = _derived_config(source, family=family, lane=lane, variant=variant)
    config["experiment"].update({"evidence_status": "authorized_formal_component_confirmation", "held_out_test_authorized": False})
    config["training"].update({
        "seed": int(seed), "max_epochs": 200, "early_stopping_patience": 0,
        "epoch_100_continue_gate": False, "required_stop_epoch": 200,
        "require_terminal_checkpoint": True, "convergence_window": [181, 200],
        "convergence_metric": "validation_bands_2_10_mre",
        "convergence_relative_threshold": 0.005,
    })
    return config


def freeze_seed0_screen(c0_dir: str | Path, output_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve()
    c0 = Path(c0_dir)
    if not c0.is_absolute():
        c0 = root / c0
    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=False)
    selected = _select_candidates(c0)
    derived = out / "derived_configs"; derived.mkdir()
    rows: list[dict[str, Any]] = []
    index = 0
    for family in ("fno", "tfno", "convnext", "joint"):
        lanes = ("joint",) if family == "joint" else ("binary", "graded")
        for lane in lanes:
            source_rel = SOURCES[(family, lane)]
            source = yaml.safe_load((root / source_rel).read_text(encoding="utf-8"))
            for variant in ("baseline", *selected[family]):
                case_id = f"{family}_{lane}_{variant}_s0"
                config_path = derived / f"{case_id}.yaml"
                config_path.write_text(yaml.safe_dump(_derived_config(source, family=family, lane=lane, variant=variant), sort_keys=False), encoding="utf-8")
                rows.append({"task_index": index, "case_id": case_id, "family": family, "lane": lane, "variant": variant, "model": FAMILIES[family][0], "seed": 0, "config": _portable_path(config_path, root), "run_dir": _portable_path(out / "runs" / case_id, root)})
                index += 1
    with (out / "task_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out / "selection.json").write_text(json.dumps({"status": "seed0_screen_frozen", "experiment_id": "EXP-FWD-019", "test_targets_loaded": False, "selected_candidates": selected, "c0_dir": _portable_path(c0, root), "c0_manifest_sha256": _sha256(c0 / "c0_manifest.csv"), "task_count": len(rows)}, indent=2) + "\n", encoding="utf-8")
    return out


def freeze_recovery(
    screen_dir: str | Path,
    output_dir: str | Path,
    *,
    task_indices: Sequence[int],
    repository_root: str | Path = ".",
) -> Path:
    """Freeze a non-overwriting correction slice from an existing screen manifest."""

    root = Path(repository_root).resolve()
    screen = Path(screen_dir); screen = screen if screen.is_absolute() else root / screen
    out = Path(output_dir); out = out if out.is_absolute() else root / out
    requested = tuple(int(value) for value in task_indices)
    if not requested or len(set(requested)) != len(requested) or any(value < 0 for value in requested):
        raise ValueError("task_indices must be a non-empty sequence of unique non-negative integers")
    with (screen / "task_manifest.csv").open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    by_index = {int(row["task_index"]): row for row in source_rows}
    missing = sorted(set(requested) - set(by_index))
    if missing:
        raise ValueError(f"recovery task indices are absent from the source manifest: {missing}")

    out.mkdir(parents=True, exist_ok=False)
    derived = out / "derived_configs"; derived.mkdir()
    rows: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for recovery_index, source_index in enumerate(requested):
        source_row = by_index[source_index]
        source_config = root / source_row["config"]
        replacement_config = derived / f'{source_row["case_id"]}.yaml'
        shutil.copyfile(source_config, replacement_config)
        source_hash = _sha256(source_config)
        replacement_hash = _sha256(replacement_config)
        if replacement_hash != source_hash:
            raise RuntimeError(f"recovery config hash mismatch for task {source_index}")
        replacement_run = out / "runs" / source_row["case_id"]
        rows.append({
            **{key: source_row[key] for key in ("case_id", "family", "lane", "variant", "model", "seed")},
            "task_index": recovery_index,
            "config": _portable_path(replacement_config, root),
            "run_dir": _portable_path(replacement_run, root),
        })
        rows[-1] = {key: rows[-1][key] for key in ("task_index", "case_id", "family", "lane", "variant", "model", "seed", "config", "run_dir")}
        lineage.append({
            "recovery_task_index": recovery_index,
            "source_task_index": source_index,
            "case_id": source_row["case_id"],
            "source_config": source_row["config"],
            "source_config_sha256": source_hash,
            "replacement_config": _portable_path(replacement_config, root),
            "replacement_config_sha256": replacement_hash,
            "source_run_dir": source_row["run_dir"],
            "replacement_run_dir": _portable_path(replacement_run, root),
        })
    with (out / "task_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / "source_to_replacement_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lineage[0])); writer.writeheader(); writer.writerows(lineage)
    (out / "recovery.json").write_text(json.dumps({
        "status": "correction_slice_frozen",
        "experiment_id": "EXP-FWD-019",
        "test_targets_loaded": False,
        "source_screen_dir": _portable_path(screen, root),
        "source_manifest_sha256": _sha256(screen / "task_manifest.csv"),
        "source_task_indices": list(requested),
        "task_count": len(rows),
    }, indent=2) + "\n", encoding="utf-8")
    return out


def _audit_seed0_run(row: Mapping[str, str], *, root: Path) -> dict[str, str]:
    run_dir = root / row["run_dir"]
    required = ("results.json", "metrics.json", "run.meta.json", "config.resolved.yaml", "model.pt")
    missing = [name for name in required if not (run_dir / name).is_file() or not (run_dir / name).stat().st_size]
    if missing:
        raise RuntimeError(f"seed0 run is missing required artifacts {missing}: {run_dir}")
    result = _read_json(run_dir / "results.json")
    metrics = _read_json(run_dir / "metrics.json")
    expected_status = "dual_lane_training_success" if row["lane"] == "joint" else "formal_training_success"
    if result.get("status") != expected_status or result.get("test_executed") is not False:
        raise RuntimeError(f"seed0 run is not successful validation-only evidence: {run_dir}")
    if result.get("formal_training_started") is not True or result.get("model") != row["model"]:
        raise RuntimeError(f"seed0 run identity mismatch: {run_dir}")
    result_identity = result.get("model_identity")
    if not isinstance(result_identity, Mapping) or result_identity.get("requested_model_name") != row["model"]:
        raise RuntimeError(f"seed0 result model identity mismatch: {run_dir}")
    if metrics.get("model_identity") != result_identity:
        raise RuntimeError(f"seed0 metrics/result model identity mismatch: {run_dir}")
    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    experiment = resolved.get("experiment", {})
    if (
        experiment.get("component_family") != row["family"]
        or experiment.get("component_variant") != row["variant"]
        or int(experiment.get("seed", -1)) != int(row["seed"])
    ):
        raise RuntimeError(f"seed0 resolved config identity mismatch: {run_dir}")
    return {name.replace(".", "_") + "_sha256": _sha256(run_dir / name) for name in required}


def freeze_accepted_seed0(
    screen_dir: str | Path,
    replacement_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    repository_root: str | Path = ".",
) -> Path:
    """Publish an audited seed0 manifest view without rewriting source run evidence."""

    root = Path(repository_root).resolve()
    screen = Path(screen_dir); screen = screen if screen.is_absolute() else root / screen
    replacements_paths = [Path(value) if Path(value).is_absolute() else root / value for value in replacement_dirs]
    out = Path(output_dir); out = out if out.is_absolute() else root / out
    with (screen / "task_manifest.csv").open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    source_by_index = {int(row["task_index"]): row for row in source_rows}
    if len(source_by_index) != len(source_rows):
        raise RuntimeError("source seed0 manifest contains duplicate task indices")
    replacements: dict[int, tuple[dict[str, str], Path]] = {}
    identity_fields = ("case_id", "family", "lane", "variant", "model", "seed")
    for replacement_dir in replacements_paths:
        with (replacement_dir / "task_manifest.csv").open(encoding="utf-8", newline="") as handle:
            replacement_rows = {int(row["task_index"]): row for row in csv.DictReader(handle)}
        with (replacement_dir / "source_to_replacement_manifest.csv").open(encoding="utf-8", newline="") as handle:
            lineage_rows = list(csv.DictReader(handle))
        for lineage in lineage_rows:
            source_index = int(lineage["source_task_index"])
            replacement_index = int(lineage["recovery_task_index"])
            if source_index in replacements or source_index not in source_by_index or replacement_index not in replacement_rows:
                raise RuntimeError(f"invalid or duplicate replacement lineage for source task {source_index}")
            source_row = source_by_index[source_index]
            replacement_row = replacement_rows[replacement_index]
            if any(source_row[field] != replacement_row[field] for field in identity_fields):
                raise RuntimeError(f"replacement identity mismatch for source task {source_index}")
            source_hash = _sha256(root / source_row["config"])
            replacement_hash = _sha256(root / replacement_row["config"])
            if source_hash != replacement_hash or source_hash != lineage["source_config_sha256"]:
                raise RuntimeError(f"replacement config mismatch for source task {source_index}")
            replacements[source_index] = (replacement_row, replacement_dir)

    accepted_rows: list[dict[str, str]] = []
    lineage: list[dict[str, Any]] = []
    run_paths: set[str] = set()
    for source_index in sorted(source_by_index):
        source_row = source_by_index[source_index]
        accepted_row, evidence_dir = replacements.get(source_index, (source_row, screen))
        if accepted_row["run_dir"] in run_paths:
            raise RuntimeError(f'duplicate accepted run path: {accepted_row["run_dir"]}')
        run_paths.add(accepted_row["run_dir"])
        artifact_hashes = _audit_seed0_run(accepted_row, root=root)
        accepted_rows.append({**accepted_row, "task_index": str(source_index)})
        lineage.append({
            "task_index": source_index,
            "case_id": source_row["case_id"],
            "source_kind": "replacement" if source_index in replacements else "original",
            "evidence_dir": _portable_path(evidence_dir, root),
            "source_run_dir": source_row["run_dir"],
            "accepted_run_dir": accepted_row["run_dir"],
            "source_config_sha256": _sha256(root / source_row["config"]),
            "accepted_config_sha256": _sha256(root / accepted_row["config"]),
            **artifact_hashes,
        })
    out.mkdir(parents=True, exist_ok=False)
    with (out / "task_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(accepted_rows[0])); writer.writeheader(); writer.writerows(accepted_rows)
    with (out / "source_to_replacement_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lineage[0])); writer.writeheader(); writer.writerows(lineage)
    shutil.copyfile(screen / "selection.json", out / "selection.json")
    (out / "acceptance.json").write_text(json.dumps({
        "status": "accepted_seed0_derivative",
        "experiment_id": "EXP-FWD-019",
        "test_targets_loaded": False,
        "source_screen_dir": _portable_path(screen, root),
        "source_manifest_sha256": _sha256(screen / "task_manifest.csv"),
        "replacement_dirs": [_portable_path(path, root) for path in replacements_paths],
        "replacement_task_indices": sorted(replacements),
        "task_count": len(accepted_rows),
    }, indent=2) + "\n", encoding="utf-8")
    return out


def _training_metric(run_dir: Path, lane: str) -> dict[str, float]:
    result = _read_json(run_dir / "results.json")
    metrics = _read_json(run_dir / "metrics.json")
    if result.get("test_executed") is not False or result.get("status") not in {"formal_training_success", "dual_lane_training_success"}:
        raise RuntimeError(f"seed0 run is not successful validation-only evidence: {run_dir}")
    if lane == "joint":
        return {
            "binary": float(metrics["binary_validation_bands_2_10_mre_pct"]),
            "graded": float(metrics["graded_validation_bands_2_10_mre_pct"]),
        }
    return {lane: float(metrics["bands_2_10_mre_pct"])}


def freeze_confirmation(screen_dir: str | Path, output_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve()
    screen = Path(screen_dir); screen = screen if screen.is_absolute() else root / screen
    out = Path(output_dir); out = out if out.is_absolute() else root / out
    out.mkdir(parents=True, exist_ok=False)
    selection = _read_json(screen / "selection.json")
    with (screen / "task_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {(row["family"], row["lane"], row["variant"]): row for row in rows}
    decisions: dict[str, Any] = {}
    selected_variants: dict[str, str] = {}
    for family, candidates in selection["selected_candidates"].items():
        lanes = ("joint",) if family == "joint" else ("binary", "graded")
        baseline_values: dict[str, float] = {}
        for lane in lanes:
            row = by_key[(family, lane, "baseline")]
            baseline_values.update(_training_metric(root / row["run_dir"], lane))
        candidate_records = []
        for candidate in candidates:
            values: dict[str, float] = {}
            for lane in lanes:
                row = by_key[(family, lane, candidate)]
                values.update(_training_metric(root / row["run_dir"], lane))
            deltas = {lane: (values[lane] - baseline_values[lane]) / baseline_values[lane] for lane in ("binary", "graded")}
            passed = (deltas["binary"] >= 0.02 and deltas["graded"] >= 0.02) or (
                max(deltas.values()) >= 0.05 and min(deltas.values()) >= -0.01
            )
            candidate_records.append({"variant": candidate, "metrics": values, "relative_degradation": deltas, "passed": passed, "mean_relative_degradation": sum(deltas.values()) / 2.0})
        passed_records = [record for record in candidate_records if record["passed"]]
        if passed_records:
            winner = max(passed_records, key=lambda record: (record["mean_relative_degradation"], record["variant"]))
            selected_variants[family] = str(winner["variant"])
        decisions[family] = {"baseline": baseline_values, "candidates": candidate_records, "selected": selected_variants.get(family), "status": "admitted" if family in selected_variants else "stopped_at_seed0"}
    derived = out / "derived_configs"; derived.mkdir()
    confirmation_rows: list[dict[str, Any]] = []
    index = 0
    for family, variant in selected_variants.items():
        lanes = ("joint",) if family == "joint" else ("binary", "graded")
        for lane in lanes:
            source_rel = SOURCES[(family, lane)]
            source = yaml.safe_load((root / source_rel).read_text(encoding="utf-8"))
            for arm in ("baseline", variant):
                for seed in (0, 1, 2):
                    case_id = f"{family}_{lane}_{arm}_s{seed}_e200"
                    config_path = derived / f"{case_id}.yaml"
                    config_path.write_text(yaml.safe_dump(_confirmation_config(source, family=family, lane=lane, variant=arm, seed=seed), sort_keys=False), encoding="utf-8")
                    confirmation_rows.append({"task_index": index, "case_id": case_id, "family": family, "lane": lane, "variant": arm, "model": FAMILIES[family][0], "seed": seed, "config": _portable_path(config_path, root), "run_dir": _portable_path(out / "runs" / case_id, root)})
                    index += 1
    if confirmation_rows:
        with (out / "task_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(confirmation_rows[0])); writer.writeheader(); writer.writerows(confirmation_rows)
    else:
        (out / "task_manifest.csv").write_text("task_index,case_id,family,lane,variant,model,seed,config,run_dir\n", encoding="utf-8")
    (out / "candidate_gate.json").write_text(json.dumps({"status": "confirmation_matrix_frozen" if confirmation_rows else "all_families_stopped_at_seed0", "experiment_id": "EXP-FWD-019", "test_targets_loaded": False, "screen_dir": _portable_path(screen, root), "screen_manifest_sha256": _sha256(screen / "task_manifest.csv"), "decisions": decisions, "selected_variants": selected_variants, "task_count": len(confirmation_rows)}, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--c0-dir"); source.add_argument("--screen-dir"); source.add_argument("--recovery-screen-dir"); source.add_argument("--accept-screen-dir")
    parser.add_argument("--task-indices")
    parser.add_argument("--replacement-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args(argv)
    if args.accept_screen_dir:
        if not args.replacement_dir:
            parser.error("at least one --replacement-dir is required with --accept-screen-dir")
        result = freeze_accepted_seed0(args.accept_screen_dir, args.replacement_dir, args.output_dir, repository_root=args.repository_root)
    elif args.recovery_screen_dir:
        if not args.task_indices:
            parser.error("--task-indices is required with --recovery-screen-dir")
        indices = tuple(int(value) for value in args.task_indices.split(","))
        result = freeze_recovery(args.recovery_screen_dir, args.output_dir, task_indices=indices, repository_root=args.repository_root)
    else:
        if args.task_indices:
            parser.error("--task-indices is only valid with --recovery-screen-dir")
        operation = freeze_seed0_screen if args.c0_dir else freeze_confirmation
        result = operation(args.c0_dir or args.screen_dir, args.output_dir, repository_root=args.repository_root)
    print(result)


if __name__ == "__main__":
    main()


__all__ = ["freeze_accepted_seed0", "freeze_confirmation", "freeze_recovery", "freeze_seed0_screen"]
