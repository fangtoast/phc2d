"""Run the frozen full-cell representation and cross-lane evaluation matrix.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from phc2d.forward.full_cell import _FullCellH5Dataset, _evaluate
from phc2d.forward.full_cell_wave import MODELS, SEEDS
from phc2d.models import build_model
from phc2d.runs import (
    RUN_LOG_NAME,
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    validate_run_artifacts,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)
from phc2d.training import DeviceConfig, resolve_device

REPRESENTATIONS = ("continuous", "t03", "t04", "t05", "t06", "t07")
THRESHOLDS = {"continuous": None, "t03": 0.3, "t04": 0.4, "t05": 0.5, "t06": 0.6, "t07": 0.7}
WAVE_LABELS = ("binary",) + REPRESENTATIONS


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _parse_wave_args(values: Sequence[str], root: Path) -> dict[str, Path]:
    waves: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--wave values must use LABEL=MANIFEST")
        label, raw_path = value.split("=", 1)
        if label not in WAVE_LABELS:
            raise ValueError(f"unknown wave label: {label}")
        path = Path(raw_path)
        waves[label] = path if path.is_absolute() else root / path
    if set(waves) != set(WAVE_LABELS):
        raise ValueError(f"cross evaluation requires wave labels {list(WAVE_LABELS)}")
    return waves


def planned_cross_evaluation_cells() -> list[tuple[str, str, str, int]]:
    """Return the frozen 576 source-target cells before any execution."""

    cells: list[tuple[str, str, str, int]] = []
    for source in WAVE_LABELS:
        targets = REPRESENTATIONS if source == "binary" else REPRESENTATIONS + ("binary",)
        for model in MODELS:
            for seed in SEEDS:
                for target in targets:
                    cells.append((source, target, model, seed))
    return cells


def _load_wave_runs(waves: Mapping[str, Path], root: Path) -> dict[tuple[str, str, int], Path]:
    runs: dict[tuple[str, str, int], Path] = {}
    for label, manifest in waves.items():
        rows = _read_csv(manifest)
        expected = {(model, seed) for model in MODELS for seed in SEEDS}
        actual = {(row["model"], int(row["seed"])) for row in rows}
        if len(rows) != 12 or actual != expected:
            raise ValueError(f"wave {label} is not a complete 4-model x 3-seed manifest")
        for row in rows:
            run = Path(row["run_dir"])
            if not run.is_absolute():
                run = root / run
            results = json.loads((run / "results.json").read_text(encoding="utf-8"))
            if results.get("status") != "formal_training_success" or results.get("test_executed"):
                raise RuntimeError(f"wave {label} run {row['model']}/seed{row['seed']} is not validation-frozen")
            runs[(label, row["model"], int(row["seed"]))] = run
    return runs


def _target_contract(label: str, *, binary_dataset: Path, graded_dataset: Path) -> dict[str, Any]:
    import h5py
    import numpy as np

    dataset = binary_dataset if label == "binary" else graded_dataset
    with h5py.File(dataset, "r") as handle:
        indices = np.asarray(handle["idx_test"], dtype=np.int64)
        grid_shape = tuple(int(v) for v in handle["material_weight_grid_64"].shape[1:])
    return {
        "path": dataset,
        "indices": indices,
        "grid_field": "material_weight_grid_64",
        "grid_shape": grid_shape,
        "sampling": "continuous" if label in {"binary", "continuous"} else "deterministic_threshold",
        "threshold": THRESHOLDS.get(label),
    }


def _plot_graded_heatmaps(out: Path, summary_rows: list[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for axis, model in zip(axes.flat, MODELS):
        matrix = np.full((len(REPRESENTATIONS), len(REPRESENTATIONS)), np.nan, dtype=float)
        for row in summary_rows:
            if row["model"] == model and row["source_representation"] in REPRESENTATIONS and row["target_representation"] in REPRESENTATIONS:
                i = REPRESENTATIONS.index(str(row["source_representation"]))
                j = REPRESENTATIONS.index(str(row["target_representation"]))
                matrix[i, j] = float(row["mre_mean_pct"])
        image = axis.imshow(matrix, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(REPRESENTATIONS)), REPRESENTATIONS, rotation=35)
        axis.set_yticks(range(len(REPRESENTATIONS)), REPRESENTATIONS)
        axis.set_xlabel("test representation")
        axis.set_ylabel("train representation")
        axis.set_title(model)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(out / "figures" / f"graded_6x6_cross_evaluation.{suffix}", dpi=180)
    plt.close(figure)


def run_full_cell_cross_evaluation(
    wave_args: Sequence[str],
    *,
    binary_dataset: str | Path,
    graded_dataset: str | Path,
    output_dir: str | Path,
    repository_root: str | Path = ".",
    device: str = "cuda:0",
) -> Path:
    """Execute all 576 frozen cells and write metrics without saving predictions."""

    root = Path(repository_root).resolve()
    waves = _parse_wave_args(wave_args, root)
    runs = _load_wave_runs(waves, root)
    binary_path = Path(binary_dataset)
    graded_path = Path(graded_dataset)
    if not binary_path.is_absolute():
        binary_path = root / binary_path
    if not graded_path.is_absolute():
        graded_path = root / graded_path
    out = make_run_dir("full_cell_forward_cross_evaluation_v1", run_dir=output_dir)
    ensure_figures_dir(out)
    logger = setup_logging(
        log_file=out / RUN_LOG_NAME,
        run_meta={
            "script": "python -m phc2d.forward.full_cell_cross_eval",
            "validation_boundary": "post_validation_freeze_test_diagnostics",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
        },
    )
    resolved = {
        "waves": {key: str(value.relative_to(root)) for key, value in waves.items()},
        "binary_dataset": str(binary_path.relative_to(root)),
        "graded_dataset": str(graded_path.relative_to(root)),
        "device": device,
        "planned_cells": 576,
        "prediction_persistence": False,
        "selection_influence": False,
    }
    write_resolved_config(out, yaml.safe_dump(resolved, sort_keys=False))

    import torch
    from torch.utils.data import DataLoader

    device_info = resolve_device(
        DeviceConfig(device=device, require_accelerator=device.startswith("cuda"), precision="fp32", amp_enabled=False, tf32_enabled=False, num_workers=4, pin_memory=True),
        torch_module=torch,
    )
    target_contracts = {
        label: _target_contract(label, binary_dataset=binary_path, graded_dataset=graded_path)
        for label in WAVE_LABELS
    }
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    cells = planned_cross_evaluation_cells()
    for cell_index, (source, target, model_name, seed) in enumerate(cells, start=1):
        run = runs[(source, model_name, seed)]
        checkpoint_path = run / "model.pt"
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device_info.torch_device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device_info.torch_device)
        model = build_model(
            model_name,
            input_shape=tuple(int(v) for v in checkpoint["input_shape"]),
            output_shape=tuple(int(v) for v in checkpoint["output_shape"]),
            **dict(checkpoint["model_params"]),
        ).to(device_info.torch_device)
        model.load_state_dict(checkpoint["state_dict"])
        mean = checkpoint["target_mean"].to(device_info.torch_device)
        std = checkpoint["target_std"].to(device_info.torch_device)
        target_cfg = target_contracts[target]
        layout = "flat" if model_name in {"mlp", "res_mlp_wide"} else "image"
        dataset = _FullCellH5Dataset(
            target_cfg["path"],
            target_cfg["indices"],
            grid_field=target_cfg["grid_field"],
            layout=layout,
            sampling=target_cfg["sampling"],
            threshold=target_cfg["threshold"],
            target_mean=mean.detach().cpu().numpy(),
            target_std=std.detach().cpu().numpy(),
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        metrics = _evaluate(model, loader, device=device_info.torch_device, mean=mean, std=std, torch=torch)
        if not math.isfinite(float(metrics["bands_2_10_mre_pct"])):
            raise RuntimeError(f"non-finite cross-evaluation cell {source}->{target}/{model_name}/seed{seed}")
        rows.append(
            {
                "source_lane": "binary" if source == "binary" else "graded",
                "source_representation": source,
                "target_lane": "binary" if target == "binary" else "graded",
                "target_representation": target,
                "model": model_name,
                "seed": seed,
                "mre_pct": metrics["bands_2_10_mre_pct"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "inference_time_s": metrics["eval_time_s"],
                "source_normalization_reused": True,
                "selection_influence": False,
            }
        )
        if cell_index % 24 == 0:
            _write_csv(out / "history.csv", rows)
            logger.info("completed cross-evaluation cells %d/%d", cell_index, len(cells))
    _write_csv(out / "history.csv", rows)
    _write_csv(out / "cross_evaluation_cells.csv", rows)

    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["source_representation"]), str(row["target_representation"]), str(row["model"]))
        grouped.setdefault(key, []).append(float(row["mre_pct"]))
    summary_rows = [
        {
            "source_representation": key[0],
            "target_representation": key[1],
            "model": key[2],
            "mre_mean_pct": statistics.mean(values),
            "mre_std_pct": statistics.stdev(values),
            "seed_count": len(values),
            "selection_influence": False,
        }
        for key, values in sorted(grouped.items())
    ]
    _write_csv(out / "cross_evaluation_summary.csv", summary_rows)
    _plot_graded_heatmaps(out, summary_rows)
    wall_time = time.perf_counter() - started
    metrics_payload = {
        "planned_cells": 576,
        "completed_cells": len(rows),
        "failed_cells": 0,
        "wall_time_total_s": wall_time,
        "inference_time_total_s": sum(float(row["inference_time_s"]) for row in rows),
    }
    write_metrics(out, metrics_payload)
    write_results(
        out,
        {
            "status": "formal_cross_evaluation_success",
            "test_split_executed": True,
            "selection_influence": False,
            "prediction_arrays_saved": False,
            "completed_cells": len(rows),
            "next_gate": "freeze_three_validation_winners",
        },
    )
    write_run_meta(out, {**device_info.metadata(), "planned_cells": 576, "completed_cells": len(rows)})
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"cross-evaluation artifact contract incomplete: {missing}")
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", action="append", required=True)
    parser.add_argument("--binary-dataset", type=Path, required=True)
    parser.add_argument("--graded-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    print(
        run_full_cell_cross_evaluation(
            args.wave,
            binary_dataset=args.binary_dataset,
            graded_dataset=args.graded_dataset,
            output_dir=args.output_dir,
            repository_root=args.repository_root,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["planned_cross_evaluation_cells", "run_full_cell_cross_evaluation"]
