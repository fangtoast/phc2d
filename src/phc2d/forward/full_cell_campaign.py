"""Freeze validation-only winners from the seven full-cell main waves.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from phc2d.forward.full_cell_cross_eval import REPRESENTATIONS, THRESHOLDS, WAVE_LABELS
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


def _parse_aggregation_args(values: Sequence[str], root: Path) -> dict[str, Path]:
    aggregations: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--aggregation values must use LABEL=DIRECTORY")
        label, raw_path = value.split("=", 1)
        if label not in WAVE_LABELS:
            raise ValueError(f"unknown aggregation label: {label}")
        path = Path(raw_path)
        aggregations[label] = path if path.is_absolute() else root / path
    if set(aggregations) != set(WAVE_LABELS):
        raise ValueError(f"winner freeze requires labels {list(WAVE_LABELS)}")
    return aggregations


def freeze_full_cell_main_winners(
    aggregation_args: Sequence[str],
    *,
    output_dir: str | Path,
    repository_root: str | Path = ".",
) -> Path:
    """Combine 84 successful runs and freeze three validation-only arms."""

    root = Path(repository_root).resolve()
    aggregations = _parse_aggregation_args(aggregation_args, root)
    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    all_runs: list[dict[str, Any]] = []
    leaderboard_rows: list[dict[str, Any]] = []
    wave_meta: dict[str, dict[str, Any]] = {}
    for label in WAVE_LABELS:
        directory = aggregations[label]
        meta = json.loads((directory / "wave.meta.json").read_text(encoding="utf-8"))
        if meta.get("status") != "formal_wave_aggregation_success" or int(meta.get("completed_runs", 0)) != 12:
            raise RuntimeError(f"wave {label} is not a complete formal aggregation")
        if meta.get("test_executed") or meta.get("test_driven_selection"):
            raise RuntimeError(f"wave {label} violated validation-only selection")
        wave_meta[label] = dict(meta)
        run_rows = _read_csv(directory / "run_manifest.csv")
        if len(run_rows) != 12:
            raise RuntimeError(f"wave {label} run manifest has {len(run_rows)} rows")
        for row in run_rows:
            all_runs.append({"wave": label, **row})
        for row in _read_csv(directory / "leaderboard.csv"):
            leaderboard_rows.append(
                {
                    "wave": label,
                    "lane": "binary" if label == "binary" else "graded",
                    "representation": label,
                    "threshold": THRESHOLDS.get(label),
                    "model": row["model"],
                    "validation_mre_mean_pct": float(row["validation_mre_mean_pct"]),
                    "validation_mre_std_pct": float(row["validation_mre_std_pct"]),
                    "success_count": int(row["success_count"]),
                    "selection_split": "validation",
                    "test_driven_selection": False,
                }
            )
    if len(all_runs) != 84 or len(leaderboard_rows) != 28:
        raise RuntimeError("main matrix must contain 84 runs and 28 model-representation rows")

    binary_winner = min(
        (row for row in leaderboard_rows if row["wave"] == "binary"),
        key=lambda row: float(row["validation_mre_mean_pct"]),
    )
    continuous_winner = min(
        (row for row in leaderboard_rows if row["wave"] == "continuous"),
        key=lambda row: float(row["validation_mre_mean_pct"]),
    )
    threshold_winner = min(
        (row for row in leaderboard_rows if row["wave"] in REPRESENTATIONS[1:]),
        key=lambda row: float(row["validation_mre_mean_pct"]),
    )
    winners = [binary_winner, continuous_winner, threshold_winner]
    selection_rows = [
        {
            "lane": row["lane"],
            "representation": row["representation"],
            "config": wave_meta[str(row["wave"])]["config_path"],
            "model": row["model"],
            "threshold": "" if row["threshold"] is None else row["threshold"],
            "validation_mre_mean_pct": row["validation_mre_mean_pct"],
            "validation_mre_std_pct": row["validation_mre_std_pct"],
            "selection_split": "validation",
            "test_driven_selection": False,
        }
        for row in winners
    ]
    _write_csv(out / "main_matrix_manifest.csv", all_runs)
    _write_csv(
        out / "main_matrix_leaderboard.csv",
        sorted(leaderboard_rows, key=lambda row: (str(row["wave"]), float(row["validation_mre_mean_pct"]))),
    )
    _write_csv(out / "winner_selection.csv", selection_rows)
    write_json(
        out / "campaign.meta.json",
        {
            "status": "main_84_training_complete_validation_winners_frozen",
            "planned_runs": 84,
            "completed_runs": 84,
            "failed_runs": 0,
            "missing_runs": 0,
            "wave_count": 7,
            "selection_split": "validation",
            "test_executed_during_selection": False,
            "winner_count": 3,
            "winners": selection_rows,
            "next_gate": "resolution_ablation_18_job_admission_probe",
        },
    )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregation", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    print(
        freeze_full_cell_main_winners(
            args.aggregation,
            output_dir=args.output_dir,
            repository_root=args.repository_root,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["freeze_full_cell_main_winners"]
