"""Freeze and gate the EXP-FWD-020 six-domain global-local bridge campaign.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


VARIANTS = ("baseline", "candidate")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("manifest cannot be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("EXP-FWD-020 controller config must be a mapping")
    return dict(value)


def _portable(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _runner_config(controller: Mapping[str, Any], domain: str, variant: str, seed: int, epochs: int) -> dict[str, Any]:
    dataset = dict(controller["datasets"][domain])
    params = {**dict(controller["model"]["shared_params"]), **dict(controller["model"]["paired_variants"][variant])}
    lane = "binary" if domain.endswith("binary") else "graded"
    return {
        "experiment": {"experiment_id": "EXP-FWD-020", "family": "full_cell_global_local_bridge", "lane": lane, "domain": domain, "variant": variant, "evidence_status": "validation_only_g1" if epochs == 20 else "g2_confirmation", "held_out_test_authorized": False},
        "dataset": {"path": dataset.get("n56_path", dataset["path"]), "grid_field": dataset["grid_field"], "expected_dataset_sha256": dataset["sha256"]},
        "representation": {"source": dataset["grid_field"], "sampling": "continuous", "threshold": None, "resolution": 64, "physical_equivalence": True, "label_semantics": f"{domain}_Wh"},
        "models": {controller["model"]["name"]: params},
        "training": {"train_size": 40000 if domain.startswith("f50") else 8000, "train_subset_salt": f"exp-fwd-020-{domain}-v1", "max_epochs": epochs, "required_stop_epoch": 200 if epochs == 200 else None, "early_stopping_patience": 0, "epoch_100_continue_gate": False, "require_terminal_checkpoint": epochs == 200, "convergence_window": [181, 200] if epochs == 200 else None, "convergence_metric": "validation_bands_2_10_mre" if epochs == 200 else None, "convergence_relative_threshold": 0.005 if epochs == 200 else None, "optimizer": controller["training"]["optimizer"], "weight_decay": controller["training"]["weight_decay"], "learning_rates": {controller["model"]["name"]: controller["training"]["learning_rate"]}, "micro_batch_size": controller["training"]["micro_batch_size"], "effective_batch_size": controller["training"]["effective_batch_size"], "loss": controller["training"]["loss"], "checkpoint_selection": "validation_bands_2_10_mre"},
        "runtime": {"device": "cuda:0", "require_accelerator": True, "precision": "fp32", "amp_enabled": False, "tf32_enabled": False, "num_workers": 4, "pin_memory": True, "non_blocking_transfer": True},
        "output": {"runs_dir": "runs"},
    }


def freeze_g1(config_path: str | Path, output_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve()
    controller_path = Path(config_path); controller_path = controller_path if controller_path.is_absolute() else root / controller_path
    out = Path(output_dir); out = out if out.is_absolute() else root / out
    out.mkdir(parents=True, exist_ok=False)
    controller = _load(controller_path)
    if not bool(controller["authorization"]["granted"]):
        raise RuntimeError("remote campaign authorization is not granted")
    derived = out / "derived_configs"; derived.mkdir()
    rows: list[dict[str, Any]] = []
    for index, domain in enumerate(controller["datasets"]):
        for variant in VARIANTS:
            case_id = f"{domain}_{variant}_s0"
            config = derived / f"{case_id}.yaml"
            config.write_text(yaml.safe_dump(_runner_config(controller, domain, variant, 0, 20), sort_keys=False), encoding="utf-8")
            rows.append({"task_index": len(rows), "case_id": case_id, "domain": domain, "lane": "binary" if domain.endswith("binary") else "graded", "variant": variant, "seed": 0, "config": _portable(config, root), "run_dir": _portable(out / "runs" / case_id, root)})
    _write_csv(out / "task_manifest.csv", rows)
    (out / "freeze.json").write_text(json.dumps({"experiment_id": "EXP-FWD-020", "stage": "G1", "test_targets_loaded": False, "task_count": len(rows), "array_limit": 8, "controller_sha256": _sha256(controller_path)}, indent=2) + "\n", encoding="utf-8")
    return out


def freeze_g2(config_path: str | Path, g1_dir: str | Path, output_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve(); controller = _load(root / config_path)
    g1 = Path(g1_dir); g1 = g1 if g1.is_absolute() else root / g1
    out = Path(output_dir); out = out if out.is_absolute() else root / out; out.mkdir(parents=True, exist_ok=False)
    with (g1 / "task_manifest.csv").open(encoding="utf-8", newline="") as handle: source = list(csv.DictReader(handle))
    by_case = {row["case_id"]: row for row in source}
    passed: list[str] = []
    decisions: dict[str, Any] = {}
    for domain in controller["datasets"]:
        base, candidate = (by_case[f"{domain}_{variant}_s0"] for variant in VARIANTS)
        outcomes = []
        for row in (base, candidate):
            run = root / row["run_dir"]
            required = ("run.log", "run.meta.json", "config.resolved.yaml", "metrics.json", "results.json", "history.csv", "model.pt")
            if any(not (run / name).is_file() for name in required):
                raise RuntimeError(f"G1 evidence incomplete: {row['case_id']}")
            result = json.loads((run / "results.json").read_text(encoding="utf-8")); metric = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            if result.get("test_executed") or result.get("selection_split") != "validation":
                raise RuntimeError(f"G1 validation boundary invalid: {row['case_id']}")
            value = float(metric["best_validation_bands_2_10_mre_pct"])
            if not value > 0.0:
                raise RuntimeError(f"G1 metric is not finite positive: {row['case_id']}")
            outcomes.append(value)
        relative = 100.0 * (outcomes[0] - outcomes[1]) / outcomes[0]
        decision = relative >= float(controller["protocol"]["seed0_screen"]["minimum_relative_improvement_pct_each_lane"])
        decisions[domain] = {"baseline_mre_pct": outcomes[0], "candidate_mre_pct": outcomes[1], "relative_improvement_pct": relative, "passed": decision}
        if decision: passed.append(domain)
    derived = out / "derived_configs"; derived.mkdir(); rows: list[dict[str, Any]] = []
    for domain in passed:
        for variant in VARIANTS:
            for seed in (0, 1, 2):
                case_id = f"{domain}_{variant}_s{seed}"; config = derived / f"{case_id}.yaml"
                config.write_text(yaml.safe_dump(_runner_config(controller, domain, variant, seed, 200), sort_keys=False), encoding="utf-8")
                rows.append({"task_index": len(rows), "case_id": case_id, "domain": domain, "lane": "binary" if domain.endswith("binary") else "graded", "variant": variant, "seed": seed, "config": _portable(config, root), "run_dir": _portable(out / "runs" / case_id, root)})
    if rows: _write_csv(out / "task_manifest.csv", rows)
    else: (out / "task_manifest.csv").write_text("task_index,case_id,domain,lane,variant,seed,config,run_dir\n", encoding="utf-8")
    (out / "candidate_gate.json").write_text(json.dumps({"experiment_id": "EXP-FWD-020", "stage": "G2", "test_targets_loaded": False, "decisions": decisions, "passed_domains": passed, "task_count": len(rows)}, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze-g1", "freeze-g2"):
        item = sub.add_parser(name); item.add_argument("--config", required=True); item.add_argument("--output-dir", required=True); item.add_argument("--repository-root", default=".")
        if name == "freeze-g2": item.add_argument("--g1-dir", required=True)
    args = parser.parse_args(argv)
    result = freeze_g1(args.config, args.output_dir, repository_root=args.repository_root) if args.command == "freeze-g1" else freeze_g2(args.config, args.g1_dir, args.output_dir, repository_root=args.repository_root)
    print(result)


__all__ = ["freeze_g1", "freeze_g2"]


if __name__ == "__main__": main()
