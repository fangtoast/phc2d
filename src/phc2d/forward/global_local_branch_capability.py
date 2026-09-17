"""Freeze, admit, train and summarize independent Global/Local capability.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.data.full_cell_mixed import file_sha256, materialize_mixed_validation

MODEL = "global_local_band_operator_v1"
VARIANTS = ("full", "g_only", "l_only", "g_only_cap", "l_only_cap")


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def parameter_controls(params: dict[str, Any], input_shape: Any, output_shape: Any,
                       search: dict[str, int], tolerance: float) -> list[dict[str, Any]]:
    """Choose nearest parameter counts using width only, without GPU allocation."""
    import torch
    from phc2d.models import build_model

    def count(branches: list[str], width: int) -> int:
        with torch.device("meta"):
            model = build_model(MODEL, input_shape=input_shape, output_shape=output_shape,
                                **{**params, "width": width, "enabled_branches": branches})
        return sum(p.numel() for p in model.parameters())

    width = int(params["width"])
    full = count(["spectral", "local"], width)
    rows = [{"variant": "full", "width": width, "parameter_count": full, "enabled_branches": ["spectral", "local"]}]
    for branch, variant in (("spectral", "g_only"), ("local", "l_only")):
        rows.append({"variant": variant, "width": width, "parameter_count": count([branch], width), "enabled_branches": [branch]})
        candidates = [(w, count([branch], w)) for w in range(search["min"], search["max"] + 1, search["step"])
                      if w % int(params["norm_groups"]) == 0]
        chosen_width, n = min(candidates, key=lambda pair: (abs(pair[1] - full), pair[0]))
        if abs(n / full - 1) > tolerance:
            raise RuntimeError(f"parameter-count control unavailable: {variant}")
        rows.append({"variant": variant + "_cap", "width": chosen_width, "parameter_count": n, "enabled_branches": [branch]})
    return sorted(rows, key=lambda row: VARIANTS.index(row["variant"]))


def freeze(config_path: str | Path, campaign: str | Path, root: str | Path = ".") -> Path:
    import h5py

    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    config_path = Path(config_path).resolve()
    contract = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if contract["experiment_id"] != "EXP-FWD-022" or tuple(contract["variants"]) != VARIANTS or contract["seeds"] != [0, 1, 2]:
        raise ValueError("capability matrix identity mismatch")
    source = yaml.safe_load((root / contract["source_controller"]).read_text(encoding="utf-8"))
    params = {**source["model"]["shared_params"], "local_kernel": contract["local_kernel"]}
    first_domain = next(iter(contract["domains_by_lane"].values()))[0]
    first_spec = source["datasets"][first_domain]
    with h5py.File(root / first_spec["path"], "r") as h:
        counts = parameter_controls(params, h[first_spec["grid_field"]].shape[1:], h["Wh"].shape[1:],
                                    contract["width_search"], contract["parameter_tolerance"])
    # Check feasibility before creating any campaign or copying expensive data.
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign / "derived_configs").mkdir()
    (campaign / "mixed_data").mkdir()
    mixed, rows = {}, []
    for lane, domains in contract["domains_by_lane"].items():
        specs = {d: {**source["datasets"][d], "path": str(root / source["datasets"][d]["path"])} for d in domains}
        mix = materialize_mixed_validation(specs, campaign / "mixed_data" / f"{lane}.h5",
                                          train_per_domain=contract["split"]["train_per_domain"],
                                          validation_per_domain=contract["split"]["validation_per_domain"])
        mixed[lane] = mix
        selected = parameter_controls(params, mix["grid_shape"], mix["target_shape"], contract["width_search"], contract["parameter_tolerance"])
        if counts is not None and selected != counts:
            raise RuntimeError("lane parameter control mismatch")
        counts = selected
        for variant in selected:
            for seed in contract["seeds"]:
                case = f"{lane}_{variant['variant']}_s{seed}"
                runner = {
                    "experiment": {"experiment_id": "EXP-FWD-022", "lane": lane, "variant": variant["variant"],
                                   "evidence_status": "independent_mixed_validation_capability", "held_out_test_authorized": False},
                    "dataset": {"path": mix["path"], "grid_field": "input_grid", "expected_dataset_sha256": mix["sha256"]},
                    "representation": {"source": "input_grid", "resolution": mix["grid_shape"][0], "sampling": "continuous",
                                       "threshold": None, "physical_equivalence": True},
                    "models": {MODEL: {**params, "width": variant["width"], "enabled_branches": variant["enabled_branches"]}},
                    "training": {**contract["training"], "train_size": mix["n_train"], "optimizer": "adamw",
                                 "loss": "normalized_full_matrix_mse", "learning_rates": {MODEL: contract["training"]["learning_rate"]},
                                 "required_stop_epoch": 200, "require_terminal_checkpoint": True, "epoch_100_continue_gate": False,
                                 "checkpoint_selection": "validation_bands_2_10_mre", "convergence_metric": "validation_bands_2_10_mre"},
                    "runtime": {"device": "cuda:0", "require_accelerator": True, "precision": "fp32", "amp_enabled": False,
                                "tf32_enabled": False, "num_workers": 4, "pin_memory": True},
                    "output": {"runs_dir": "runs"},
                }
                path = campaign / "derived_configs" / f"{case}.yaml"
                path.write_text(yaml.safe_dump(runner, sort_keys=False), encoding="utf-8")
                rows.append({"task_index": len(rows), "case_id": case, "lane": lane, "variant": variant["variant"], "seed": seed,
                             "config": str(path), "config_sha256": file_sha256(path), "run_dir": str(campaign / "runs" / case),
                             "parameter_count": variant["parameter_count"], "dataset_sha256": mix["sha256"]})
    if len(rows) != 30:
        raise RuntimeError("capability contract requires 30 tasks")
    _csv(campaign / "task_manifest.csv", rows)
    _json(campaign / "mixed_data_manifest.json", mixed)
    _json(campaign / "parameter_controls.json", counts)
    (campaign / "config.resolved.yaml").write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    _json(campaign / "freeze.json", {"experiment_id": "EXP-FWD-022", "task_count": 30, "contract_sha256": file_sha256(config_path),
                                    "manifest_sha256": file_sha256(campaign / "task_manifest.csv"),
                                    "execution_start_git_commit": os.environ.get("PHC2D_DEPLOYED_GIT_COMMIT"), "test_executed": False})
    _json(campaign / "campaign_status.json", {"status": "capability_frozen_pending_preflight", "test_executed": False})
    return campaign


def task(campaign: Path, index: int) -> dict[str, str]:
    snapshot = json.loads((campaign / "freeze.json").read_text())
    if file_sha256(campaign / "task_manifest.csv") != snapshot["manifest_sha256"]:
        raise RuntimeError("training manifest hash drift")
    rows = _rows(campaign / "task_manifest.csv")
    selected = [r for r in rows if int(r["task_index"]) == index]
    if len(selected) != 1:
        raise ValueError("task must resolve exactly once")
    row = selected[0]
    if row["variant"] not in VARIANTS or row["case_id"] != f"{row['lane']}_{row['variant']}_s{row['seed']}":
        raise RuntimeError("task identity mismatch")
    if file_sha256(row["config"]) != row["config_sha256"]:
        raise RuntimeError("task config hash drift")
    return row


def run(campaign: Path, index: int, probe: bool = False) -> Path:
    from phc2d.forward.full_cell import run_full_cell_forward

    row = task(campaign, index)
    config = yaml.safe_load(Path(row["config"]).read_text())
    if file_sha256(config["dataset"]["path"]) != row["dataset_sha256"]:
        raise RuntimeError("mixed dataset hash drift")
    if not probe and json.loads((campaign / "campaign_status.json").read_text())["status"] not in ("capability_preflight_passed", "capability_training_submitted"):
        raise RuntimeError("training requires preflight pass")
    output = campaign / "admission" / row["case_id"] if probe else Path(row["run_dir"])
    if output.exists():
        raise FileExistsError(output)
    result = run_full_cell_forward(row["config"], model_name=MODEL, seed=int(row["seed"]), run_dir=output, admission_probe=probe)
    metrics = json.loads((result / "metrics.json").read_text())
    terminal = json.loads((result / "results.json").read_text())
    if metrics["parameter_count"] != int(row["parameter_count"]) or terminal["test_executed"] or terminal["selection_split"] != "validation":
        raise RuntimeError("training parameter/validation identity mismatch")
    if probe and terminal["status"] != "full_cell_forward_admission_probe_pass":
        raise RuntimeError("admission probe failed")
    if not probe and (terminal["stop_epoch"] != 200 or terminal["history_epoch_count"] != 200 or not (result / "epoch200.pt").is_file()):
        raise RuntimeError("fixed 200-epoch task incomplete")
    return result


def preflight(campaign: Path) -> Path:
    rows = _rows(campaign / "task_manifest.csv")
    completed = []
    for row in rows:
        if row["seed"] == "0":
            output = run(campaign, int(row["task_index"]), probe=True)
            completed.append({"case_id": row["case_id"], "run_dir": str(output), "status": "passed"})
    if len(completed) != 10:
        raise RuntimeError("preflight requires both lanes and all five structures")
    _csv(campaign / "preflight_manifest.csv", completed)
    _json(campaign / "campaign_status.json", {"status": "capability_preflight_passed", "test_executed": False})
    return campaign


def _validation_samples(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    import h5py
    import torch
    from phc2d.models import build_model

    run_dir = Path(row["run_dir"])
    result = json.loads((run_dir / "results.json").read_text())
    if result["test_executed"] or result["stop_epoch"] != 200:
        raise RuntimeError("capability evaluation requires completed validation-only training")
    checkpoint = run_dir / "model.pt"
    if file_sha256(checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash drift")
    payload = torch.load(checkpoint, map_location="cuda:0", weights_only=True)
    config = yaml.safe_load(Path(row["config"]).read_text())
    if file_sha256(config["dataset"]["path"]) != row["dataset_sha256"]:
        raise RuntimeError("mixed dataset hash drift before evaluation")
    model = build_model(MODEL, input_shape=payload["input_shape"], output_shape=payload["output_shape"], **payload["model_params"]).cuda().eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    with h5py.File(config["dataset"]["path"], "r") as h:
        indices = np.asarray(h["idx_val"], dtype=np.int64)
        grids = np.asarray(h["input_grid"][indices], dtype=np.float32)
        targets = np.asarray(h["Wh"][indices], dtype=np.float32)
        domain = np.asarray(h["source_domain_index"][indices], dtype=np.int64)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), 8):
            x = torch.as_tensor(grids[start:start + 8, None], device="cuda:0")
            y = model(x) * payload["target_std"] + payload["target_mean"]
            predictions.append(y.cpu().numpy())
    prediction = np.concatenate(predictions)
    sample = np.abs(prediction[..., 1:] - targets[..., 1:]) / np.maximum(np.abs(targets[..., 1:]), 1e-8)
    sample = sample.mean(axis=(1, 2)) * 100
    if not np.isfinite(sample).all():
        raise RuntimeError("nonfinite capability sample error")
    np.savez_compressed(run_dir / "validation_sample_errors.npz", mre_pct=sample, source_domain_index=domain, mixed_indices=indices)
    return sample, domain


def finalize(campaign: Path) -> Path:
    from phc2d.forward.global_local_bridge_evaluation import paired_bootstrap

    rows = _rows(campaign / "task_manifest.csv")
    samples, summary, comparisons, benefits = {}, [], [], []
    for row in rows:
        task(campaign, int(row["task_index"]))
        sample, domain = _validation_samples(row)
        samples[(row["lane"], row["variant"], row["seed"])] = sample
        result = json.loads((Path(row["run_dir"]) / "results.json").read_text())
        summary.append({"case_id": row["case_id"], "lane": row["lane"], "variant": row["variant"], "seed": row["seed"],
                        "mixed_mre_pct": float(sample.mean()), "k5_mre_pct": float(sample[domain == 0].mean()),
                        "k20_mre_pct": float(sample[domain == 1].mean()), "parameter_count": row["parameter_count"],
                        "convergence_passed": result["convergence"]["passed"], "checkpoint_sha256": result["checkpoint_sha256"]})
    for lane in ("binary", "graded"):
        for seed in ("0", "1", "2"):
            full = samples[(lane, "full", seed)]
            for variant in VARIANTS[1:]:
                comparisons.append({"lane": lane, "seed": seed, "variant": variant,
                                    **paired_bootstrap(samples[(lane, variant, seed)], full, key=f"capability:{lane}:{seed}:{variant}")})
            for suffix in ("", "_cap"):
                eg = float(samples[(lane, "g_only" + suffix, seed)].mean())
                el = float(samples[(lane, "l_only" + suffix, seed)].mean())
                benefits.append({"lane": lane, "seed": seed, "parameter_control": bool(suffix), "b_gl_pp": min(eg, el) - float(full.mean())})
    _csv(campaign / "capability_metrics.csv", summary)
    _csv(campaign / "paired_bootstrap.csv", comparisons)
    _csv(campaign / "combination_benefits.csv", benefits)
    _json(campaign / "results.json", {"status": "capability_complete_pending_human_review", "training_tasks": len(rows),
                                     "test_executed": False, "physical_claim": False, "model_promotion": False,
                                     "all_converged": all(r["convergence_passed"] for r in summary)})
    _json(campaign / "campaign_status.json", {"status": "capability_complete_pending_human_review", "test_executed": False})
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "preflight", "train", "finalize"))
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    if args.stage == "freeze":
        result = freeze(args.config, args.campaign_dir, args.repository_root)
    elif args.stage == "preflight":
        result = preflight(args.campaign_dir)
    elif args.stage == "train":
        result = run(args.campaign_dir, args.task_index)
    else:
        result = finalize(args.campaign_dir)
    print(result)


if __name__ == "__main__":
    main()
