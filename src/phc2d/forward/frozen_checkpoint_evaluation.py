"""Evaluate frozen Full-Wh checkpoints on training or external datasets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from phc2d.forward.budget_confirmation import file_digest
from phc2d.forward.full_cell import _FullCellH5Dataset, _read_contract
from phc2d.forward.full_cell_upgrade_test import _as_numpy, _checkpoint_payload, _evaluate_lane, _write_csv


def _indices_and_families(path: Path, split: str) -> tuple[np.ndarray, list[str]]:
    import h5py

    with h5py.File(path, "r") as handle:
        indices = np.asarray(handle[f"idx_{split}"], dtype=np.int64)
        if len(indices) != len(np.unique(indices)) or np.any(indices < 0) or np.any(indices >= len(handle["Wh"])):
            raise ValueError("invalid evaluation split indices")
        if "structure_json" in handle:
            families = [json.loads(value)["parameters"]["shape"] for value in handle["structure_json"][indices]]
        else:
            families = ["native"] * len(indices)
    return indices, families


def _external_contract(path: Path, expected_sha256: str, reference: dict, grid_field: str) -> dict:
    import h5py

    actual = file_digest(path)
    if actual != expected_sha256:
        raise ValueError("external dataset SHA-256 mismatch")
    with h5py.File(path, "r") as handle:
        if grid_field not in handle or "Wh" not in handle or "k_points" not in handle:
            raise ValueError("external dataset is missing model inputs or labels")
        if tuple(handle[grid_field].shape[1:]) != tuple(reference["grid_shape"]):
            raise ValueError("external grid shape differs from checkpoint input")
        if tuple(handle["Wh"].shape[1:]) != tuple(reference["target_shape"]):
            raise ValueError("external target shape differs from checkpoint output")
        k_points = np.asarray(handle["k_points"], dtype=np.float32)
        if hashlib.sha256(k_points.tobytes()).hexdigest() != reference["k_points_sha256"]:
            raise ValueError("external k-point order differs from checkpoint training data")
    return {"path": str(path), "sha256": actual, "grid_field": grid_field}


def evaluate_case(row: dict[str, str], *, split: str, output: Path,
                  external_path: Path | None = None, external_sha256: str | None = None,
                  external_grid_field: str = "material_area_fraction_grid_64") -> dict:
    """Evaluate one checkpoint; create a new result directory without changing training artifacts."""
    import torch
    from torch.utils.data import DataLoader

    from phc2d.models import build_model, get_model_spec
    from phc2d.training import DeviceConfig, resolve_device

    if split not in {"train", "test"} or (external_path is None) != (external_sha256 is None):
        raise ValueError("split or external dataset contract is invalid")
    if output.exists():
        raise FileExistsError(output)
    checkpoint = Path(row["source_run_dir"]) / "model.pt"
    expected_checkpoint = row["checkpoint_sha256"]
    if file_digest(checkpoint) != expected_checkpoint:
        raise ValueError("checkpoint SHA-256 mismatch")
    payload = _checkpoint_payload(checkpoint, device="cpu", torch=torch)
    if payload["model_name"] != row["model"]:
        raise ValueError("checkpoint model identity mismatch")
    resolved = payload["resolved_config"]
    source_domain = resolved["experiment"]["source_domain"]
    if not row["task_id"].startswith(source_domain + "_"):
        raise ValueError("checkpoint source domain mismatch")
    contract = _read_contract(resolved)
    mean = _as_numpy(payload["target_mean"])
    std = _as_numpy(payload["target_std"])
    if not np.allclose(mean, contract["target_mean"], rtol=1e-5, atol=1e-6) or not np.allclose(std, contract["target_std"], rtol=1e-5, atol=1e-6):
        raise ValueError("checkpoint normalizer mismatch")
    if payload.get("k_points_sha256") and payload["k_points_sha256"] != contract["k_points_sha256"]:
        raise ValueError("checkpoint k-point identity mismatch")
    if split == "train":
        indices = contract["selected_train"]
        path = contract["path"]
        grid_field = contract["grid_field"]
        data_identity = {"path": str(path), "sha256": contract["dataset_sha256"],
                         "grid_field": grid_field, "idx_hash": contract["train_subset_hash"]}
    else:
        assert external_path is not None and external_sha256 is not None
        data_identity = _external_contract(external_path, external_sha256, contract, external_grid_field)
        path = external_path
        grid_field = external_grid_field
        indices, _ = _indices_and_families(path, "test")
        data_identity["idx_hash"] = hashlib.sha256(indices.tobytes()).hexdigest()
    if len(indices) == 0:
        raise ValueError("empty evaluation split")
    _, families = _indices_and_families(path, split) if split == "test" else (indices, ["native"] * len(indices))
    device_info = resolve_device(DeviceConfig.from_mapping(resolved["runtime"]), torch_module=torch)
    if not device_info.is_cuda:
        raise RuntimeError("formal checkpoint evaluation requires CUDA")
    device = device_info.torch_device
    spec = get_model_spec(row["model"])
    kwargs = dict(payload["model_params"])
    if spec.accepts_k_points:
        kwargs["k_points"] = payload.get("k_points")
    model = build_model(spec.name, input_shape=tuple(payload["input_shape"]),
                        output_shape=tuple(payload["output_shape"]), **kwargs).to(device)
    model.load_state_dict(payload["state_dict"])
    k_points = torch.as_tensor(payload["k_points"], dtype=torch.float32, device=device) if spec.accepts_k_points else None
    representation = resolved["representation"]
    dataset = _FullCellH5Dataset(path, indices, grid_field=grid_field,
                                 layout=representation["layout"], sampling=representation.get("sampling", "continuous"),
                                 threshold=representation.get("threshold"), target_mean=mean, target_std=std)
    loader = DataLoader(dataset, batch_size=int(resolved["training"].get("micro_batch_size", 64)),
                        shuffle=False, num_workers=device_info.num_workers, pin_memory=device_info.pin_memory)
    lane = row["lane"]
    metrics, bands, samples = _evaluate_lane(model, loader, lane=lane, lane_index=None,
                                             mean=torch.as_tensor(mean, device=device), std=torch.as_tensor(std, device=device),
                                             model_spec=spec, k_points=k_points, sample_indices=indices,
                                             device=device, torch=torch)
    for sample, family in zip(samples, families, strict=True):
        sample["family"] = family
    output.mkdir(parents=True)
    _write_csv(output / "per_sample_metrics.csv", samples)
    _write_csv(output / "per_band_metrics.csv", bands)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    result = {"case_id": row["task_id"], "source_domain": source_domain,
              "model": row["model"], "seed": int(row["seed"]), "lane": lane,
              "evaluation_role": "training_overlap" if split == "train" else "external_geometry_transfer",
              "checkpoint_sha256": expected_checkpoint, "data": data_identity,
              "sample_count": len(indices), "bands_1_10_mre_pct": metrics["test_full_matrix_mre_pct"],
              "bands_2_10_mre_pct": metrics["test_bands_2_10_mre_pct"],
              "label_resolution_status": "new_geometry_resolution_not_verified" if split == "test" else "original_training_data",
              "input_distribution": "binary_area_fraction" if split == "test" else "native_training",
              "cross_lane_input": bool(split == "test" and lane == "G")}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "config.resolved.yaml").write_text(json.dumps({"evaluation": result, "training_config_from_checkpoint": resolved}, indent=2) + "\n", encoding="utf-8")
    (output / "run.meta.json").write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                                       "script": "phc2d.forward.frozen_checkpoint_evaluation",
                                                       "device_name": device_info.device_name}, indent=2) + "\n", encoding="utf-8")
    (output / "history.csv").write_text("evaluation_only,no_training_history\ntrue,true\n", encoding="utf-8")
    (output / "run.log").write_text("Frozen checkpoint evaluation completed.\n", encoding="utf-8")
    (output / "figures").mkdir()
    return result


def summarize_campaign(manifest_path: Path, output_root: Path) -> None:
    """Require every frozen cell, then report seed cells and equal-family transfer means."""
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 54 or len({row["task_id"] for row in manifest}) != 54:
        raise ValueError("expected 54 unique frozen checkpoint identities")
    cells = []
    family_cells = []
    for dataset in ("train", "dat007_12000", "dat008_24000"):
        for row in manifest:
            out = output_root / dataset / row["task_id"]
            result = json.loads((out / "results.json").read_text(encoding="utf-8"))
            if result["checkpoint_sha256"] != row["checkpoint_sha256"] or result["sample_count"] <= 0:
                raise ValueError("evaluation identity or count mismatch")
            expected_count = 8000 if dataset == "train" else (12000 if dataset == "dat007_12000" else 24000)
            if result["sample_count"] != expected_count:
                raise ValueError("evaluation sample count mismatch")
            with (out / "per_sample_metrics.csv").open(encoding="utf-8", newline="") as handle:
                samples = list(csv.DictReader(handle))
            if len(samples) != expected_count or not np.isfinite([float(s["full_matrix_mre_pct"]) for s in samples]).all():
                raise ValueError("incomplete or nonfinite per-sample metrics")
            by_family: dict[str, list[float]] = {}
            for sample in samples:
                by_family.setdefault(sample["family"], []).append(float(sample["full_matrix_mre_pct"]))
            if len(by_family) != (1 if dataset == "train" else (6 if dataset == "dat007_12000" else 12)):
                raise ValueError("unexpected geometry family count")
            for family, values in by_family.items():
                family_cells.append({"dataset": dataset, "source_domain": result["source_domain"],
                                     "model": result["model"], "seed": result["seed"], "family": family,
                                     "sample_count": len(values), "bands_1_10_mre_pct": float(np.mean(values)),
                                     "p95_sample_mre_pct": float(np.percentile(values, 95))})
            cells.append({"dataset": dataset, "source_domain": result["source_domain"],
                          "model": result["model"], "seed": result["seed"], "lane": result["lane"],
                          "sample_count": expected_count, "bands_1_10_mre_pct": result["bands_1_10_mre_pct"],
                          "bands_2_10_mre_pct": result["bands_2_10_mre_pct"],
                          "equal_family_bands_1_10_mre_pct": float(np.mean([np.mean(v) for v in by_family.values()])),
                          "cross_lane_input": result["cross_lane_input"],
                          "checkpoint_sha256": result["checkpoint_sha256"]})
    _write_csv(output_root / "evaluation_cells.csv", cells)
    _write_csv(output_root / "family_cells.csv", family_cells)
    summaries = []
    for dataset, domain, model in sorted({(r["dataset"], r["source_domain"], r["model"]) for r in cells}):
        group = [r for r in cells if (r["dataset"], r["source_domain"], r["model"]) == (dataset, domain, model)]
        if {r["seed"] for r in group} != {0, 1, 2}:
            raise ValueError("three-seed summary identity mismatch")
        values = [r["bands_1_10_mre_pct"] for r in group]
        summaries.append({"dataset": dataset, "source_domain": domain, "model": model,
                          "seed_count": 3, "bands_1_10_mre_pct_mean": float(np.mean(values)),
                          "bands_1_10_mre_pct_seed_sd": float(np.std(values, ddof=1)),
                          "cross_lane_input": group[0]["cross_lane_input"]})
    _write_csv(output_root / "evaluation_summary.csv", summaries)
    (output_root / "results.json").write_text(json.dumps({"status": "frozen_evaluation_complete",
                                                       "checkpoint_count": 54, "evaluation_cell_count": len(cells),
                                                       "family_cell_count": len(family_cells),
                                                       "training_overlap_separate": True,
                                                       "new_geometry_resolution_verified": False,
                                                       "physical_claim": False}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--split", choices=["train", "test"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--external-path", type=Path)
    parser.add_argument("--external-sha256")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize_campaign(args.manifest, args.output_root)
        return
    if args.task_index is None or args.split is None:
        parser.error("--task-index and --split are required for evaluation")
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(row["task_index"]) == args.task_index]
    if len(matches) != 1:
        raise ValueError("task index is not unique")
    evaluate_case(matches[0], split=args.split, output=args.output_root / matches[0]["task_id"],
                  external_path=args.external_path, external_sha256=args.external_sha256)


if __name__ == "__main__":
    main()
