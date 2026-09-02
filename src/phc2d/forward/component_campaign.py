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
    source.add_argument("--c0-dir"); source.add_argument("--screen-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args(argv)
    operation = freeze_seed0_screen if args.c0_dir else freeze_confirmation
    print(operation(args.c0_dir or args.screen_dir, args.output_dir, repository_root=args.repository_root))


if __name__ == "__main__":
    main()


__all__ = ["freeze_confirmation", "freeze_seed0_screen"]
