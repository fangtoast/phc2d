"""Aggregate 18 full-cell resolution runs and freeze final validation winners.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from phc2d.runs import write_json


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


def aggregate_full_cell_resolution_ablation(
    selection_csv: str | Path,
    ablation_manifest: str | Path,
    *,
    main_matrix_manifest: str | Path,
    output_dir: str | Path,
    repository_root: str | Path = ".",
) -> Path:
    """Validate 84+18 runs and choose one resolution per frozen arm using validation only."""

    root = Path(repository_root).resolve()

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    selections = _read_csv(resolve(selection_csv))
    manifest_rows = _read_csv(resolve(ablation_manifest))
    main_runs = _read_csv(resolve(main_matrix_manifest))
    if len(selections) != 3 or len(manifest_rows) != 18 or len(main_runs) != 84:
        raise RuntimeError("resolution acceptance requires 3 selections, 18 ablation runs, and 84 main runs")
    out = resolve(output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], list[float]] = {}
    run_lookup: dict[tuple[str, str, int, int], str] = {}
    for row in manifest_rows:
        run = resolve(row["run_dir"])
        results = json.loads((run / "results.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        required = ["run.log", "run.meta.json", "config.resolved.yaml", "results.json", "metrics.json", "history.csv", "model.pt", "figures/training_curve.png", "figures/training_curve.pdf"]
        missing = [name for name in required if not (run / name).is_file()]
        if missing:
            raise RuntimeError(f"ablation run {row['job_id']} is missing {missing}")
        if results.get("status") != "formal_training_success" or results.get("test_executed"):
            raise RuntimeError(f"ablation run {row['job_id']} is not validation-frozen success")
        value = float(metrics["best_validation_bands_2_10_mre_pct"])
        if not math.isfinite(value):
            raise RuntimeError(f"ablation run {row['job_id']} has non-finite validation metric")
        representation = row["representation"]
        model = row["model"]
        resolution = int(row["resolution"])
        seed = int(row["seed"])
        grouped.setdefault((representation, model, resolution), []).append(value)
        run_lookup[(representation, model, resolution, seed)] = row["run_dir"]
        run_rows.append({**row, "status": results["status"], "best_validation_mre_pct": value, "checkpoint_sha256": results["checkpoint_sha256"]})

    summary_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for selection in selections:
        representation = selection["representation"]
        model = selection["model"]
        candidates: list[dict[str, Any]] = []
        for resolution in (32, 128):
            values = grouped.get((representation, model, resolution), [])
            if len(values) != 3:
                raise RuntimeError(f"missing three seeds for {representation}/{model}/r{resolution}")
            candidates.append(
                {
                    "lane": selection["lane"],
                    "representation": representation,
                    "model": model,
                    "threshold": selection["threshold"],
                    "resolution": resolution,
                    "validation_mre_mean_pct": statistics.mean(values),
                    "validation_mre_std_pct": statistics.stdev(values),
                    "source": "resolution_ablation",
                }
            )
        candidates.append(
            {
                "lane": selection["lane"],
                "representation": representation,
                "model": model,
                "threshold": selection["threshold"],
                "resolution": 64,
                "validation_mre_mean_pct": float(selection["validation_mre_mean_pct"]),
                "validation_mre_std_pct": float(selection["validation_mre_std_pct"]),
                "source": "main_matrix",
            }
        )
        summary_rows.extend(candidates)
        winner = min(candidates, key=lambda row: float(row["validation_mre_mean_pct"]))
        final_rows.append(
            {
                **winner,
                "config": selection["config"],
                "selection_split": "validation",
                "test_driven_selection": False,
            }
        )
    _write_csv(out / "ablation_run_manifest.csv", run_rows)
    _write_csv(out / "resolution_summary.csv", summary_rows)
    _write_csv(out / "final_resolution_selection.csv", final_rows)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    for selection in selections:
        relevant = sorted(
            [row for row in summary_rows if row["representation"] == selection["representation"]],
            key=lambda row: int(row["resolution"]),
        )
        axis.errorbar(
            [int(row["resolution"]) for row in relevant],
            [float(row["validation_mre_mean_pct"]) for row in relevant],
            yerr=[float(row["validation_mre_std_pct"]) for row in relevant],
            marker="o",
            capsize=4,
            label=f"{selection['representation']} / {selection['model']}",
        )
    axis.set_xlabel("input resolution")
    axis.set_ylabel("best validation bands 2-10 MRE (%)")
    axis.set_xticks([32, 64, 128])
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(out / "figures" / f"resolution_ablation.{suffix}", dpi=180)
    plt.close(figure)

    write_json(
        out / "acceptance.meta.json",
        {
            "status": "all_102_forward_training_runs_complete_final_resolutions_frozen",
            "main_runs": 84,
            "resolution_runs": 18,
            "total_training_runs": 102,
            "failed_runs": 0,
            "missing_runs": 0,
            "selection_split": "validation",
            "test_executed_during_selection": False,
            "final_winners": final_rows,
            "next_gate": "final_three_configuration_test_evaluation",
        },
    )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--ablation-manifest", type=Path, required=True)
    parser.add_argument("--main-matrix-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    print(
        aggregate_full_cell_resolution_ablation(
            args.selection_csv,
            args.ablation_manifest,
            main_matrix_manifest=args.main_matrix_manifest,
            output_dir=args.output_dir,
            repository_root=args.repository_root,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["aggregate_full_cell_resolution_ablation"]
