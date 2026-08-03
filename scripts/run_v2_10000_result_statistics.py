"""
Generate the V2 10k formal-result statistical analysis from existing artifacts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phc2d.data.lift import lift_rho_to_grid


EXPERIMENT_ID = "v2_10000_result_statistics"
DEFAULT_SEED = 20260706
BOOTSTRAP_REPEATS = 1000
DENOM_FLOOR = 1e-6
PALETTE = {
    "blue": "#2364AA",
    "green": "#3A7D44",
    "red": "#B4433F",
    "gold": "#C18F2E",
    "teal": "#218380",
    "purple": "#6F4E7C",
    "gray": "#5F6872",
    "ink": "#24292F",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
    else:
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def numeric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = to_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def bootstrap_ci(values: Sequence[float], rng: np.random.Generator, repeats: int = BOOTSTRAP_REPEATS) -> dict[str, float | int | None]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None, "bootstrap_repeats": int(repeats)}
    means = np.empty(repeats, dtype=float)
    for i in range(repeats):
        means[i] = np.mean(rng.choice(arr, size=arr.size, replace=True))
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "bootstrap_repeats": int(repeats),
    }


def mre_pct(y_true: np.ndarray, y_pred: np.ndarray, band_slice: slice | None = None) -> np.ndarray:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if band_slice is not None:
        true = true[..., band_slice]
        pred = pred[..., band_slice]
    denom = np.maximum(np.abs(true), DENOM_FLOOR)
    return 100.0 * np.mean(np.abs(pred - true) / denom, axis=tuple(range(1, true.ndim)))


def file_sha256(path: Path, max_bytes: int | None = None) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        remaining = max_bytes
        while True:
            chunk_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if chunk_size <= 0:
                break
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return hasher.hexdigest()


def artifact_info(path: Path, root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": rel(path, root),
        "exists": path.exists(),
        "status": "available" if path.exists() else "unavailable",
    }
    if not path.exists():
        return info
    info["size_bytes"] = int(path.stat().st_size)
    info["sha256"] = file_sha256(path, max_bytes=None if path.stat().st_size < 10_000_000 else 4_000_000)
    try:
        if path.suffix == ".json":
            obj = read_json(path)
            info["kind"] = "json"
            if isinstance(obj, dict):
                info["top_level_keys"] = list(obj)[:50]
            elif isinstance(obj, list):
                info["list_length"] = len(obj)
        elif path.suffix == ".csv":
            with path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                header = next(reader, [])
                row_count = sum(1 for _ in reader)
            info.update({"kind": "csv", "columns": header, "row_count": int(row_count)})
        elif path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                first = next(fh, "").strip()
                line_count = (1 if first else 0) + sum(1 for _ in fh)
            info.update({"kind": "jsonl", "line_count": int(line_count)})
            if first:
                info["first_keys"] = list(json.loads(first))[:50]
        elif path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
            info.update({"kind": "npy", "shape": [int(v) for v in arr.shape], "dtype": str(arr.dtype)})
    except Exception as exc:  # pragma: no cover - artifact audit should not crash on one bad file.
        info["inspect_error"] = f"{type(exc).__name__}: {exc}"
        info["status"] = "inspect_error"
    return info


def build_artifact_index(paths: dict[str, Path], root: Path) -> dict[str, Any]:
    expected = [
        "metrics.json",
        "results.json",
        "history.csv",
        "leaderboard.csv",
        "per_band_metrics.csv",
        "per_k_metrics.csv",
        "protocol_metrics.csv",
        "ranked_candidates.csv",
        "per_target_replay_metrics.csv",
        "per_band_replay_metrics.csv",
        "per_target_band_replay_metrics.csv",
        "replay_candidates.jsonl",
        "target_bands.npy",
        "surrogate_pred_bands.npy",
        "mpb_replay_bands.npy",
        "candidates.json",
        "frozen_surrogates.json",
    ]
    runs: dict[str, Any] = {}
    for label, run_dir in paths.items():
        records = []
        for name in expected:
            records.append({"artifact": name, **artifact_info(run_dir / name, root)})
        runs[label] = {
            "run_dir": rel(run_dir, root),
            "exists": run_dir.exists(),
            "artifacts": records,
        }
    return {"runs": runs}


def group_mean_std(rows: Sequence[dict[str, str]], group_keys: Sequence[str], metric: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = to_float(row.get(metric))
        if value is not None:
            buckets[tuple(row.get(k, "") for k in group_keys)].append(value)
    out: list[dict[str, Any]] = []
    for key, values in sorted(buckets.items()):
        stats = summarize(values)
        row = {group_keys[i]: key[i] for i in range(len(group_keys))}
        row.update({f"{metric}_{k}": v for k, v in stats.items()})
        out.append(row)
    return out


def selected_forward_case(manifest: dict[str, Any], root: Path) -> tuple[str, Path, Path]:
    case_id = manifest["selected_full_wh_surrogate"]["case_id"]
    run_dir = root / manifest["forward_run"]["run_dir"]
    eval_dir = run_dir / "cases" / case_id / "eval"
    return case_id, eval_dir / "target_bands.npy", eval_dir / "surrogate_pred_bands.npy"


def sample_without_replacement(n_available: int, n_requested: int, rng: np.random.Generator) -> np.ndarray:
    if n_available <= 0:
        return np.asarray([], dtype=int)
    return np.sort(rng.choice(n_available, size=min(n_available, n_requested), replace=False))


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    svg = path.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def plot_forward_sample_panel(target: np.ndarray, pred: np.ndarray, sample_idx: np.ndarray, out_path: Path) -> list[dict[str, Any]]:
    sampled_true = target[sample_idx]
    sampled_pred = pred[sample_idx]
    sample_mre = mre_pct(sampled_true, sampled_pred, slice(1, None))
    per_band_mre_bands_2_10 = 100.0 * np.mean(
        np.abs(sampled_pred[..., 1:] - sampled_true[..., 1:])
        / np.maximum(np.abs(sampled_true[..., 1:]), DENOM_FLOOR),
        axis=(0, 1),
    )
    first = int(sample_idx[0]) if sample_idx.size else 0
    error_heatmap = np.abs(pred[first] - target[first])

    fig = plt.figure(figsize=(11.2, 6.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])
    ax0 = fig.add_subplot(grid[0, 0])
    ax0.hist(sample_mre, bins=18, color=PALETTE["blue"], alpha=0.82)
    ax0.axvline(float(np.mean(sample_mre)), color=PALETTE["red"], linewidth=1.4, label=f"mean={np.mean(sample_mre):.3f}%")
    ax0.set_title("Forward sampled MRE, bands 2-10")
    ax0.set_xlabel("MRE (%)")
    ax0.set_ylabel("sample count")
    ax0.legend(frameon=False)

    ax1 = fig.add_subplot(grid[0, 1])
    ax1.bar(np.arange(2, per_band_mre_bands_2_10.size + 2), per_band_mre_bands_2_10, color=PALETTE["teal"])
    ax1.set_title("Forward per-band MRE on sampled eval rows, bands 2-10")
    ax1.set_xlabel("band")
    ax1.set_ylabel("MRE (%)")

    ax2 = fig.add_subplot(grid[1, 0])
    x = np.arange(target.shape[1])
    for band in range(target.shape[2]):
        alpha = 0.85 if band > 0 else 0.35
        ax2.plot(x, target[first, :, band], color=PALETTE["ink"], linewidth=0.85, alpha=alpha)
        ax2.plot(x, pred[first, :, band], color=PALETTE["gold"], linewidth=0.85, linestyle="--", alpha=alpha)
    ax2.set_title(f"Example eval sample {first}: target vs surrogate")
    ax2.set_xlabel("k index")
    ax2.set_ylabel("frequency")

    ax3 = fig.add_subplot(grid[1, 1])
    im = ax3.imshow(error_heatmap.T, aspect="auto", origin="lower", cmap="magma")
    ax3.set_title("Example absolute error heatmap")
    ax3.set_xlabel("k index")
    ax3.set_ylabel("band")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    fig.suptitle("Forward full-Wh surrogate 100-sample panel", fontsize=13)
    save_figure(fig, out_path)

    return [
        {
            "sample_source": "forward_eval_band_arrays",
            "sample_position": int(pos),
            "mre_bands_2_10_pct": float(err),
        }
        for pos, err in zip(sample_idx, sample_mre)
    ]


def plot_query_per_k(per_k_rows: Sequence[dict[str, str]], sampled_rows: Sequence[dict[str, Any]], out_path: Path) -> None:
    protocols = sorted({row.get("protocol", "") for row in per_k_rows})
    colors = {protocol: list(PALETTE.values())[i % len(PALETTE)] for i, protocol in enumerate(protocols)}
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for protocol in protocols:
        rows = [row for row in per_k_rows if row.get("protocol") == protocol]
        x = np.asarray(numeric_values(rows, "k_index"), dtype=float)
        y = np.asarray(numeric_values(rows, "query_mre_bands_2_10_pct"), dtype=float)
        if x.size and y.size:
            ax.scatter(x, y, s=18, alpha=0.28, color=colors[protocol], label=protocol)
    sx = np.asarray([row["k_index"] for row in sampled_rows], dtype=float)
    sy = np.asarray([row["query_mre_bands_2_10_pct"] for row in sampled_rows], dtype=float)
    if sx.size:
        ax.scatter(sx, sy, s=36, facecolors="none", edgecolors=PALETTE["red"], linewidth=1.0, label="sampled metric rows")
    ax.set_title("Query surrogate per-k metric distribution (per-query predictions unavailable)")
    ax.set_xlabel("k index")
    ax.set_ylabel("MRE bands 2-10 (%)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, out_path)


def flatten_candidate_records(candidates_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for target in candidates_payload.get("targets", []):
        selected = target.get("selected_indices", [])
        errors = target.get("surrogate_metrics", {}).get("selected_reconstruction_errors", [])
        ranked = target.get("ranked_indices", [])
        candidates = target.get("candidates", [])
        disagreement = target.get("surrogate_metrics", {}).get("cross_surrogate_disagreement_bands_2_10")
        for rank, candidate_index in enumerate(selected, start=1):
            if not isinstance(candidate_index, int) or candidate_index < 0 or candidate_index >= len(candidates):
                continue
            rho = np.asarray(candidates[candidate_index], dtype=float)
            binary = rho >= 0.5
            try:
                grid = lift_rho_to_grid(rho)
                grid_fill = float(np.mean(grid >= 0.5))
            except Exception:
                grid_fill = float(np.mean(binary))
            records.append(
                {
                    "target_id": target.get("target_id"),
                    "target_index": int(target.get("target_index")),
                    "sample_id": target.get("sample_id"),
                    "candidate_index": int(candidate_index),
                    "candidate_rank": int(rank),
                    "ranked_position": int(ranked.index(candidate_index) + 1) if candidate_index in ranked else int(rank),
                    "surrogate_mre_bands_2_10_pct": float(errors[rank - 1] * 100.0) if rank - 1 < len(errors) else None,
                    "cross_surrogate_disagreement_bands_2_10_pct": float(disagreement * 100.0) if disagreement is not None else None,
                    "fill_fraction": float(np.mean(rho)),
                    "binary_fill_fraction": float(np.mean(binary)),
                    "grid_fill_fraction": grid_fill,
                    "rho": rho.tolist(),
                }
            )
    return records


def plot_candidate_quality(records: Sequence[dict[str, Any]], out_path: Path) -> None:
    ranks = np.asarray([r["candidate_rank"] for r in records], dtype=float)
    surrogate = np.asarray([r["surrogate_mre_bands_2_10_pct"] for r in records], dtype=float)
    disagreement = np.asarray([r["cross_surrogate_disagreement_bands_2_10_pct"] for r in records], dtype=float)
    fill = np.asarray([r["grid_fill_fraction"] for r in records], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    axes[0, 0].scatter(ranks, surrogate, c=fill, cmap="viridis", s=34)
    axes[0, 0].set_xlabel("candidate rank")
    axes[0, 0].set_ylabel("surrogate MRE b2-10 (%)")
    axes[0, 0].set_title("Selected candidate surrogate quality")

    axes[0, 1].scatter(surrogate, disagreement, c=ranks, cmap="plasma", s=34)
    axes[0, 1].set_xlabel("surrogate MRE b2-10 (%)")
    axes[0, 1].set_ylabel("cross-surrogate disagreement (%)")
    axes[0, 1].set_title("Error vs disagreement")

    axes[1, 0].hist(fill, bins=16, color=PALETTE["green"], alpha=0.82)
    axes[1, 0].set_xlabel("grid fill fraction")
    axes[1, 0].set_ylabel("candidate count")
    axes[1, 0].set_title("Geometry fill distribution")

    axes[1, 1].boxplot(
        [surrogate[ranks == rank] for rank in sorted(set(ranks))],
        tick_labels=[str(int(r)) for r in sorted(set(ranks))],
    )
    axes[1, 1].set_xlabel("candidate rank")
    axes[1, 1].set_ylabel("surrogate MRE b2-10 (%)")
    axes[1, 1].set_title("Quality by selected rank")
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.suptitle("Inverse candidate sampled quality", fontsize=13)
    save_figure(fig, out_path)


def plot_candidate_geometry(records: Sequence[dict[str, Any]], out_path: Path) -> None:
    n = min(len(records), 100)
    cols = 10
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12.0, max(3.0, rows * 1.2)))
    axes_arr = np.asarray(axes).reshape(rows, cols)
    for ax in axes_arr.flat:
        ax.set_axis_off()
    for idx, record in enumerate(records[:n]):
        ax = axes_arr.flat[idx]
        grid = lift_rho_to_grid(np.asarray(record["rho"], dtype=float))
        ax.imshow(grid >= 0.5, cmap="Greys", origin="lower", interpolation="nearest")
        ax.set_title(f"t{record['target_index']} r{record['candidate_rank']}", fontsize=6)
    fig.suptitle(f"Inverse candidate geometry grid, n={n}", fontsize=12)
    save_figure(fig, out_path)


def plot_replay_targets(target: np.ndarray, surrogate: np.ndarray, replay: np.ndarray, per_target: Sequence[dict[str, str]], out_path: Path) -> None:
    n = target.shape[0]
    cols = 5
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14.0, rows * 2.0), sharex=True)
    axes_arr = np.asarray(axes).reshape(rows, cols)
    for ax in axes_arr.flat:
        ax.set_axis_off()
    x = np.arange(target.shape[1])
    for i in range(n):
        ax = axes_arr.flat[i]
        ax.set_axis_on()
        ax.plot(x, np.mean(target[i, :, 1:], axis=1), color=PALETTE["ink"], linewidth=1.0, label="target")
        ax.plot(x, np.mean(surrogate[i, :, 1:], axis=1), color=PALETTE["gold"], linewidth=1.0, linestyle="--", label="surrogate")
        ax.plot(x, np.mean(replay[i, :, 1:], axis=1), color=PALETTE["blue"], linewidth=1.0, linestyle=":", label="MPB")
        metric = to_float(per_target[i].get("best_of_k_replay_mre_bands_2_10_pct")) if i < len(per_target) else None
        ax.set_title(f"{per_target[i].get('target_index', i) if i < len(per_target) else i} | {metric:.2f}%" if metric is not None else str(i), fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.14)
    handles, labels = axes_arr.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.002), ncol=3, frameon=False)
    fig.suptitle(f"Formal replay all targets: target vs surrogate vs MPB, n={n}", fontsize=12, y=0.992)
    fig.tight_layout(rect=[0.0, 0.035, 1.0, 0.965])
    save_figure(fig, out_path)


def plot_replay_candidate_trace(per_target: Sequence[dict[str, str]], sampled_candidates: Sequence[dict[str, Any]], out_path: Path) -> None:
    rows = list(per_target)
    x = np.arange(len(rows))
    replay = np.asarray(numeric_values(rows, "best_of_k_replay_mre_bands_2_10_pct"), dtype=float)
    surrogate = np.asarray(numeric_values(rows, "best_of_k_surrogate_mre_bands_2_10_pct"), dtype=float)
    gap = np.asarray(numeric_values(rows, "surrogate_replay_gap_bands_2_10_pct"), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=False)
    axes[0].plot(x, replay, marker="o", linewidth=1.2, label="MPB replay MRE b2-10")
    axes[0].plot(x, surrogate, marker="s", linewidth=1.2, label="surrogate MRE b2-10")
    axes[0].plot(x, gap, marker="^", linewidth=1.2, label="surrogate-replay gap")
    axes[0].set_title("Replay target-level traces (candidate-level replay bands unavailable)")
    axes[0].set_xlabel("target row")
    axes[0].set_ylabel("percent")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    ranks = [r["candidate_rank"] for r in sampled_candidates]
    fills = [r["grid_fill_fraction"] for r in sampled_candidates]
    errors = [r["surrogate_mre_bands_2_10_pct"] for r in sampled_candidates]
    sc = axes[1].scatter(ranks, errors, c=fills, cmap="viridis", s=34)
    axes[1].set_title(f"Sampled selected candidate records, n={len(sampled_candidates)}")
    axes[1].set_xlabel("candidate rank")
    axes[1].set_ylabel("surrogate MRE b2-10 (%)")
    axes[1].grid(alpha=0.2)
    fig.colorbar(sc, ax=axes[1], label="grid fill fraction")
    save_figure(fig, out_path)


def plot_baseline_comparison(rows: Sequence[dict[str, Any]], out_path: Path) -> None:
    labels = [row["label"] for row in rows]
    replay = [row["best_of_k_replay_mre_bands_2_10_pct"] for row in rows]
    gap = [row["surrogate_replay_gap_bands_2_10_pct"] for row in rows]
    target_counts = [row["targets"] for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    width = 0.34
    ax.bar(x - width / 2, replay, width, label="MPB replay MRE b2-10", color=PALETTE["blue"])
    ax.bar(x + width / 2, gap, width, label="surrogate-replay gap b2-10", color=PALETTE["red"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\nn={n}" for label, n in zip(labels, target_counts)], rotation=0)
    ax.set_ylabel("percent")
    ax.set_title("Baseline vs mixture real-MPB replay comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, out_path)


def plot_worst_gallery(target: np.ndarray, surrogate: np.ndarray, replay: np.ndarray, worst_rows: Sequence[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(len(worst_rows), 1, figsize=(10.5, max(4.0, len(worst_rows) * 1.45)), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    x = np.arange(target.shape[1])
    for ax, row in zip(axes_arr, worst_rows):
        pos = int(row["position"])
        ax.plot(x, np.mean(target[pos, :, 1:], axis=1), color=PALETTE["ink"], linewidth=1.1, label="target")
        ax.plot(x, np.mean(surrogate[pos, :, 1:], axis=1), color=PALETTE["gold"], linestyle="--", linewidth=1.1, label="surrogate")
        ax.plot(x, np.mean(replay[pos, :, 1:], axis=1), color=PALETTE["blue"], linestyle=":", linewidth=1.1, label="MPB")
        ax.set_ylabel(f"idx {row['target_index']}", fontsize=8)
        ax.set_title(
            f"rank {row['case_rank']} | replay={row['best_of_k_replay_mre_bands_2_10_pct']:.2f}% | gap={row['surrogate_replay_gap_bands_2_10_pct']:.2f}%",
            fontsize=8,
        )
        ax.grid(alpha=0.16)
    axes_arr[0].legend(frameon=False, ncol=3, fontsize=8)
    axes_arr[-1].set_xlabel("k index")
    fig.suptitle("Worst replay/gap target gallery, bands 2-10 mean curve", fontsize=12)
    save_figure(fig, out_path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_gap(row: dict[str, Any]) -> str:
    replay = float(row["best_of_k_replay_mre_bands_2_10_pct"])
    surrogate = float(row["best_of_k_surrogate_mre_bands_2_10_pct"])
    gap = float(row["surrogate_replay_gap_bands_2_10_pct"])
    if gap >= 5 and surrogate <= 3:
        return "surrogate_exploit_like"
    if replay >= 12 and surrogate >= 5:
        return "target_or_geometry_hard_case"
    if abs(gap) <= 2:
        return "ordinary_error_variation"
    return "mixed_or_requires_geometry_review"


def make_report(
    *,
    root: Path,
    run_dir: Path,
    report_path: Path,
    metrics: dict[str, Any],
    paths: dict[str, Path],
    samples: dict[str, Any],
    artifact_index: dict[str, Any],
    baseline_rows: Sequence[dict[str, Any]],
    forward_top: Sequence[dict[str, Any]],
    query_protocol_rows: Sequence[dict[str, Any]],
    replay_stats: dict[str, Any],
    best_rows: Sequence[dict[str, Any]],
    worst_rows: Sequence[dict[str, Any]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    unavailable = []
    for label, run_payload in artifact_index["runs"].items():
        for item in run_payload["artifacts"]:
            if item["status"] != "available":
                unavailable.append((label, item["artifact"]))

    def table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            vals = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, float):
                    vals.append(f"{val:.4g}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    header = """<!--
  V2 10k formal-result statistical analysis report.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->
"""
    lines = [
        header,
        '<a id="v2-10000-statistical-analysis"></a>',
        "",
        "# V2 10k Formal Result Statistical Analysis",
        "",
        f"- 报告日期：2026-07-06",
        f"- 统计 run：[{rel(run_dir, root)}](../../{rel(run_dir, root)})",
        "- 环境边界：本报告只复用已有 artifacts；未重新训练，未重新运行 MPB/Meep。",
        "- 统计口径：inverse headline 固定使用 bands 2-10，band1/full-band MRE 仅作为 near-zero denominator 风险说明。",
        "",
        "**目录：** [Executive Summary](#executive-summary) · [Evidence Boundary](#evidence-boundary) · [Data And Artifact Index](#data-and-artifact-index) · [Forward/Query Statistical Summary](#forwardquery-statistical-summary) · [Inverse Candidate Preflight Summary](#inverse-candidate-preflight-summary) · [Real-MPB Replay Summary](#real-mpb-replay-summary) · [Baseline Comparison](#baseline-comparison) · [100-Sample Visualization Gallery](#100-sample-visualization-gallery) · [Worst/Best Case Analysis](#worstbest-case-analysis) · [Surrogate-vs-MPB Gap Analysis](#surrogate-vs-mpb-gap-analysis) · [Residual Risks](#residual-risks) · [Next Plan](#next-plan)",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- forward eval band arrays 可抽样，实际抽样 n={samples['forward']['n']}；selected case 为 `{metrics['forward']['selected_case_id']}`。",
        f"- query 未发现 per-query prediction arrays，使用 per-k metric rows 作为替代图；per-k 抽样 n={samples['query']['metric_row_n']}，per-query sample n=0。",
        f"- repaired preflight 从 selected candidate 记录抽样 n={samples['candidate']['n']}；候选来自 40 个 target 的 selected top-M。",
        f"- formal real-MPB replay target 数为 n={samples['replay']['target_n']}，不是 100；报告和图使用全部 target。",
        f"- formal replay headline `best_of_k_replay_mre_bands_2_10_pct` = {metrics['replay']['metrics'].get('best_of_k_replay_mre_bands_2_10_pct'):.4f}%，`surrogate_replay_gap_bands_2_10_pct` = {metrics['replay']['metrics'].get('surrogate_replay_gap_bands_2_10_pct'):.4f}%。",
        f"- surrogate/replay target-level 相关性 Pearson r = {replay_stats['surrogate_vs_replay_corr']:.4f}。",
        "",
        "---",
        "",
        "## Evidence Boundary",
        "",
        "本轮只读取既有 forward/query/inverse/replay 产物和已提交报告。未启动新的 forward/query/inverse 训练；未启动新的 MPB/Meep replay。formal replay 指标来自已有 WSL `Ubuntu` + conda `meep_env` real-MPB artifacts，统计脚本本身只是 NumPy/Matplotlib 离线分析。",
        "",
        "实际读取的核心 artifacts：",
        "",
    ]
    for key in ["forward", "query", "frozen_manifest", "preflight", "mixture_replay", "old_conditional_replay", "direct_replay"]:
        lines.append(f"- **{key}：** `{rel(paths[key], root)}`")
    lines += [
        "",
        "---",
        "",
        "## Data And Artifact Index",
        "",
        f"完整索引写入 `{rel(run_dir / 'artifact_index.json', root)}`。关键缺失项没有补造，已在 index 标为 unavailable。",
        "",
        f"- unavailable 项数量：{len(unavailable)}",
        f"- sample manifest：`{rel(run_dir / 'sample_manifest.json', root)}`",
        f"- metrics summary：`{rel(run_dir / 'metrics_summary.json', root)}`",
        "",
        "---",
        "",
        "## Forward/Query Statistical Summary",
        "",
        "Forward leaderboard top rows（按 `avg_MRE_bands_2_10_pct`）：",
        "",
        table(forward_top[:5], ["rank", "case_id", "model", "input_repr", "seed", "learning_rate", "avg_MRE_bands_2_10_pct", "checkpoint_path"]),
        "",
        "Query protocol summary（来自 frozen manifest 中选定 fno2d/lr=1e-3 的 3-seed 边界）：",
        "",
        table(query_protocol_rows, ["protocol", "model", "n_seeds", "query_mre_bands_2_10_pct_mean", "query_mre_bands_2_10_pct_std"]),
        "",
        f"heldout-segment diagnostic gate pass = `{metrics['query']['heldout_segment_gate_pass']}`；该 split 保留为更难诊断，不写成通过。",
        "",
        "---",
        "",
        "## Inverse Candidate Preflight Summary",
        "",
        f"- `validity_rate` = {metrics['preflight'].get('validity_rate')}",
        f"- `unique_rate` = {metrics['preflight'].get('unique_rate')}",
        f"- `nearest_neighbor_hamming_p10` = {metrics['preflight'].get('nearest_neighbor_hamming_p10')}",
        f"- `nearest_neighbor_hamming_min` = {metrics['preflight'].get('nearest_neighbor_hamming_min')}",
        f"- `largest_duplicate_cluster_fraction` = {metrics['preflight'].get('largest_duplicate_cluster_fraction')}",
        f"- `best_of_k_surrogate_mre_bands_2_10` = {metrics['preflight'].get('best_of_k_surrogate_mre_bands_2_10')}",
        f"- `cross_surrogate_disagreement_bands_2_10` = {metrics['preflight'].get('cross_surrogate_disagreement_bands_2_10')}",
        "",
        "---",
        "",
        "## Real-MPB Replay Summary",
        "",
        "Replay target-level bands 2-10 统计与 bootstrap 置信区间：",
        "",
        table(
            [
                {
                    "metric": key,
                    **value,
                }
                for key, value in metrics["replay_bootstrap"].items()
            ],
            ["metric", "n", "mean", "ci95_low", "ci95_high", "bootstrap_repeats"],
        ),
        "",
        "---",
        "",
        "## Baseline Comparison",
        "",
        table(
            baseline_rows,
            [
                "label",
                "targets",
                "replay_top_m",
                "best_of_k_replay_mre_bands_2_10_pct",
                "best_of_k_surrogate_mre_bands_2_10_pct",
                "surrogate_replay_gap_bands_2_10_pct",
                "mpb_replay_success_rate",
            ],
        ),
        "",
        "注意：old conditional/direct baseline 与 mixture 的 target count/top-M 不同，因此本表用于公平口径提示与方向性对照，不把旧 baseline 的失败 replay 写成通过。",
        "",
        "---",
        "",
        "## 100-Sample Visualization Gallery",
        "",
        f"- `figures/forward_100_sample_error_panel.png`：forward eval arrays 抽样 n={samples['forward']['n']}。",
        f"- `figures/query_100_sample_error_vs_k.png`：query per-query samples unavailable；使用 per-k metric rows 抽样 n={samples['query']['metric_row_n']}。",
        f"- `figures/inverse_candidate_100_sample_quality.png`：candidate selected records 抽样 n={samples['candidate']['n']}。",
        f"- `figures/inverse_candidate_100_sample_geometry_grid.png`：candidate geometry grid 抽样 n={samples['candidate']['n']}。",
        f"- `figures/replay_all_targets_mpb_vs_surrogate.png`：formal replay target n={samples['replay']['target_n']}，not 100。",
        f"- `figures/replay_100_candidate_trace.png`：replay candidate-level bands unavailable；图中 replay error 使用 target-level traces，candidate panel 使用 selected candidate records n={samples['candidate']['n']}。",
        "- `figures/baseline_vs_mixture_replay_comparison.png`：baseline 与 mixture replay 对照。",
        "- `figures/worst_case_gallery.png`：worst 10 target 曲线。",
        "",
        "---",
        "",
        "## Worst/Best Case Analysis",
        "",
        "Worst 10（按 replay MRE 与 gap 的较大值排序）：",
        "",
        table(worst_rows, ["case_rank", "target_index", "best_of_k_replay_mre_bands_2_10_pct", "surrogate_replay_gap_bands_2_10_pct", "failure_label"]),
        "",
        "Best 10（按 replay MRE bands 2-10 排序）：",
        "",
        table(best_rows, ["case_rank", "target_index", "best_of_k_replay_mre_bands_2_10_pct", "surrogate_replay_gap_bands_2_10_pct", "failure_label"]),
        "",
        "---",
        "",
        "## Surrogate-vs-MPB Gap Analysis",
        "",
        f"Pearson r = {replay_stats['surrogate_vs_replay_corr']:.4f}。高 gap target 已按启发式标注为 `surrogate_exploit_like`、`target_or_geometry_hard_case`、`ordinary_error_variation` 或 `mixed_or_requires_geometry_review`；这些标签只来自已有统计与曲线形态，不替代新的 solver 诊断。",
        "",
        "band1/full-band MRE 风险：band1 在 Gamma 附近存在 near-zero denominator 放大，formal inverse headline 固定使用 bands 2-10。full-band MRE 可保留为诊断，但不作为结论排名。",
        "",
        "---",
        "",
        "## Residual Risks",
        "",
        "- query 无 per-query prediction arrays，不能画真正的 100 条 query-level 误差散点，只能用 per-k/protocol 汇总替代。",
        "- replay candidate JSONL 有 200 条 top-M 元数据，但没有每个 candidate 的 replay band arrays，因此 candidate-level replay error 不能补造。",
        "- baseline 之间 target count/top-M 不同，不能解读成严格同 target 配对统计检验。",
        "- 本报告没有新增 MPB/Meep 计算；任何进一步物理判断必须另启明确的 WSL `meep_env` replay gate。",
        "",
        "---",
        "",
        "## Next Plan",
        "",
        "唯一建议：基于本报告的 worst/high-gap target 清单，下一轮只做一个小规模、同 target、同 top-M 的 paired replay/diagnostic 设计，用来判断高 gap 是 surrogate exploit 还是 candidate geometry issue。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--report-path", type=Path, default=Path("docs/results/v2_10000_statistical_analysis_20260706.md"))
    args = parser.parse_args(argv)

    root = repo_root()
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    run_dir = args.run_dir or root / "runs" / "20260706" / f"{EXPERIMENT_ID}_{timestamp}"
    run_dir = run_dir if run_dir.is_absolute() else root / run_dir
    report_path = args.report_path if args.report_path.is_absolute() else root / args.report_path
    figures = run_dir / "figures"
    tables = run_dir / "summary_tables"
    rng = np.random.default_rng(int(args.seed))

    paths = {
        "forward": root / "runs/20260702/forward_formal_retrain_v2_10000_202607021842",
        "query": root / "runs/20260702/query_formal_retrain_v2_10000_202607022213",
        "frozen_manifest": root / "runs/20260704/frozen_surrogate_manifest_v2_10000_202607041715",
        "preflight": root / "runs/20260706/inverse_mixture_generator_v2_10000_objective_sample_separation_202607060848",
        "mixture_replay": root / "runs/20260706/inverse_mpb_replay_mixture_v2_10000_40_202607060853",
        "old_conditional_replay": root / "runs/20260630/inverse_mpb_replay_old_conditional_v2_10000_unique24_202606302055",
        "direct_replay": root / "runs/20260630/inverse_mpb_replay_v2_10000_direct_baseline_202606301709",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    log_lines = [
        f"started_at={datetime.now(timezone.utc).isoformat()}",
        "boundary=no new training; no new MPB/Meep replay",
    ]
    manifest = read_json(paths["frozen_manifest"] / "frozen_surrogates.json")
    artifact_index = build_artifact_index(paths, root)
    write_json(run_dir / "artifact_index.json", artifact_index)

    forward_rows = read_csv(paths["forward"] / "leaderboard.csv")
    forward_rows = [row for row in forward_rows if row.get("status") == "ok"]
    forward_sorted = sorted(forward_rows, key=lambda r: to_float(r.get("avg_MRE_bands_2_10_pct")) or float("inf"))
    forward_top: list[dict[str, Any]] = []
    for rank, row in enumerate(forward_sorted, start=1):
        forward_top.append(
            {
                "rank": rank,
                "case_id": row.get("case_id"),
                "model": row.get("model"),
                "input_repr": row.get("input_repr"),
                "seed": row.get("seed"),
                "learning_rate": row.get("learning_rate"),
                "avg_MRE_bands_2_10_pct": to_float(row.get("avg_MRE_bands_2_10_pct")),
                "avg_MRE_pct": to_float(row.get("avg_MRE_pct")),
                "train_time_s": to_float(row.get("train_time_s")),
                "eval_time_s": to_float(row.get("eval_time_s")),
                "checkpoint_path": row.get("checkpoint_path"),
            }
        )
    write_csv(tables / "forward_leaderboard_ranked.csv", forward_top)
    write_csv(
        tables / "forward_seed_summary.csv",
        group_mean_std(forward_rows, ["base_case_id", "model", "input_repr", "learning_rate"], "avg_MRE_bands_2_10_pct"),
    )
    per_band_forward = read_csv(paths["forward"] / "per_band_metrics.csv")
    best_case_id = manifest["selected_full_wh_surrogate"]["case_id"]
    write_csv(tables / "forward_selected_case_per_band.csv", [row for row in per_band_forward if row.get("case_id") == best_case_id])

    selected_case_id, target_path, pred_path = selected_forward_case(manifest, root)
    forward_target = np.load(target_path)
    forward_pred = np.load(pred_path)
    forward_idx = sample_without_replacement(int(forward_target.shape[0]), 100, rng)
    forward_sample_rows = plot_forward_sample_panel(forward_target, forward_pred, forward_idx, figures / "forward_100_sample_error_panel.png")

    query_history = read_csv(paths["query"] / "history.csv")
    query_leaderboard = read_csv(paths["query"] / "leaderboard.csv")
    query_per_k = read_csv(paths["query"] / "per_k_metrics.csv")
    query_sample_idx = sample_without_replacement(len(query_per_k), 100, rng)
    query_sample_rows: list[dict[str, Any]] = []
    for pos in query_sample_idx:
        row = query_per_k[int(pos)]
        query_sample_rows.append(
            {
                "sample_source": "query_per_k_metrics_substitute",
                "row_position": int(pos),
                "protocol": row.get("protocol"),
                "model": row.get("model"),
                "seed": row.get("seed"),
                "k_index": int(float(row.get("k_index", 0))),
                "query_mre_bands_2_10_pct": to_float(row.get("query_mre_bands_2_10_pct")),
            }
        )
    plot_query_per_k(query_per_k, query_sample_rows, figures / "query_100_sample_error_vs_k.png")
    query_protocol_rows: list[dict[str, Any]] = []
    boundary_map = manifest.get("query_protocol_boundaries", {})
    protocol_aliases = [
        ("unseen_structure_all_k", "unseen_structure_all_k_fno2d_lr1e-3"),
        ("unseen_structure_physical_k_interpolation", "unseen_structure_physical_k_interpolation_fno2d_lr1e-3"),
        ("unseen_structure_physical_heldout_segment", "unseen_structure_physical_heldout_segment_fno2d_lr1e-3"),
    ]
    for protocol, boundary_key in protocol_aliases:
        payload = boundary_map.get(boundary_key, {})
        query_protocol_rows.append(
            {
                "protocol": protocol,
                "model": manifest["selected_query_surrogate"]["model"],
                "n_seeds": payload.get("n"),
                "query_mre_bands_2_10_pct_mean": payload.get("mean"),
                "query_mre_bands_2_10_pct_std": payload.get("std"),
            }
        )
    write_csv(tables / "query_protocol_summary.csv", query_protocol_rows)
    write_csv(
        tables / "query_seed_summary.csv",
        group_mean_std(query_history, ["protocol", "model", "learning_rate"], "query_mre_bands_2_10_pct"),
    )
    write_csv(tables / "query_sampled_per_k_rows.csv", query_sample_rows)

    preflight_metrics = read_json(paths["preflight"] / "metrics.json")
    ranked_candidates = read_csv(paths["preflight"] / "ranked_candidates.csv")
    candidate_payload = read_json(paths["preflight"] / "candidates.json")
    candidate_records_all = flatten_candidate_records(candidate_payload)
    candidate_idx = sample_without_replacement(len(candidate_records_all), 100, rng)
    candidate_sample_rows = [candidate_records_all[int(i)] for i in candidate_idx]
    candidate_sample_light = [{k: v for k, v in row.items() if k != "rho"} for row in candidate_sample_rows]
    write_csv(tables / "inverse_candidate_sample_records.csv", candidate_sample_light)
    write_csv(tables / "inverse_preflight_target_summary.csv", ranked_candidates)
    plot_candidate_quality(candidate_sample_rows, figures / "inverse_candidate_100_sample_quality.png")
    plot_candidate_geometry(candidate_sample_rows, figures / "inverse_candidate_100_sample_geometry_grid.png")

    replay_metrics = read_json(paths["mixture_replay"] / "metrics.json")
    per_target = read_csv(paths["mixture_replay"] / "per_target_replay_metrics.csv")
    per_band_replay = read_csv(paths["mixture_replay"] / "per_band_replay_metrics.csv")
    per_target_band = read_csv(paths["mixture_replay"] / "per_target_band_replay_metrics.csv")
    replay_candidates = read_jsonl(paths["mixture_replay"] / "replay_candidates.jsonl")
    target_bands = np.load(paths["mixture_replay"] / "target_bands.npy")
    surrogate_bands = np.load(paths["mixture_replay"] / "surrogate_pred_bands.npy")
    mpb_bands = np.load(paths["mixture_replay"] / "mpb_replay_bands.npy")
    plot_replay_targets(target_bands, surrogate_bands, mpb_bands, per_target, figures / "replay_all_targets_mpb_vs_surrogate.png")
    plot_replay_candidate_trace(per_target, candidate_sample_rows, figures / "replay_100_candidate_trace.png")

    replay_analysis_rows: list[dict[str, Any]] = []
    for pos, row in enumerate(per_target):
        replay_value = to_float(row.get("best_of_k_replay_mre_bands_2_10_pct")) or 0.0
        gap_value = to_float(row.get("surrogate_replay_gap_bands_2_10_pct")) or 0.0
        surrogate_value = to_float(row.get("best_of_k_surrogate_mre_bands_2_10_pct")) or 0.0
        out = {
            "position": pos,
            "target_id": row.get("target_id"),
            "target_index": int(float(row.get("target_index", pos))),
            "best_of_k_replay_mre_bands_2_10_pct": replay_value,
            "best_of_k_surrogate_mre_bands_2_10_pct": surrogate_value,
            "surrogate_replay_gap_bands_2_10_pct": gap_value,
            "candidate_count_selected": int(float(row.get("candidate_count_selected", 0) or 0)),
        }
        out["failure_label"] = classify_gap(out)
        out["sort_score"] = max(replay_value, abs(gap_value))
        replay_analysis_rows.append(out)
    worst_rows = sorted(replay_analysis_rows, key=lambda r: r["sort_score"], reverse=True)[:10]
    best_rows = sorted(replay_analysis_rows, key=lambda r: r["best_of_k_replay_mre_bands_2_10_pct"])[:10]
    for rank, row in enumerate(worst_rows, start=1):
        row["case_rank"] = rank
    for rank, row in enumerate(best_rows, start=1):
        row["case_rank"] = rank
    write_csv(tables / "replay_target_summary.csv", replay_analysis_rows)
    write_csv(tables / "replay_per_band_summary.csv", per_band_replay)
    write_csv(tables / "replay_per_target_band_summary.csv", per_target_band)
    write_csv(tables / "replay_worst10_targets.csv", worst_rows)
    write_csv(tables / "replay_best10_targets.csv", best_rows)
    plot_worst_gallery(target_bands, surrogate_bands, mpb_bands, worst_rows, figures / "worst_case_gallery.png")

    baseline_rows: list[dict[str, Any]] = []
    for label, key in [
        ("old_conditional_baseline", "old_conditional_replay"),
        ("direct_inverse_baseline", "direct_replay"),
        ("mixture_conditional_generator", "mixture_replay"),
    ]:
        payload = read_json(paths[key] / "metrics.json")
        baseline_rows.append(
            {
                "label": label,
                "run_dir": rel(paths[key], root),
                "targets": payload.get("targets"),
                "replay_top_m": payload.get("replay_top_m"),
                "best_of_k_replay_mre_bands_2_10_pct": payload.get("best_of_k_replay_mre_bands_2_10_pct"),
                "best_of_k_surrogate_mre_bands_2_10_pct": payload.get("best_of_k_surrogate_mre_bands_2_10_pct"),
                "surrogate_replay_gap_bands_2_10_pct": payload.get("surrogate_replay_gap_bands_2_10_pct"),
                "mpb_replay_success_rate": payload.get("mpb_replay_success_rate"),
                "validation_boundary": payload.get("validation_boundary"),
            }
        )
    write_csv(tables / "baseline_comparison.csv", baseline_rows)
    plot_baseline_comparison(baseline_rows, figures / "baseline_vs_mixture_replay_comparison.png")

    replay_values = numeric_values(per_target, "best_of_k_replay_mre_bands_2_10_pct")
    surrogate_values = numeric_values(per_target, "best_of_k_surrogate_mre_bands_2_10_pct")
    gap_values = numeric_values(per_target, "surrogate_replay_gap_bands_2_10_pct")
    replay_bootstrap = {
        "best_of_k_replay_mre_bands_2_10_pct": {**summarize(replay_values), **bootstrap_ci(replay_values, rng)},
        "best_of_k_surrogate_mre_bands_2_10_pct": {**summarize(surrogate_values), **bootstrap_ci(surrogate_values, rng)},
        "surrogate_replay_gap_bands_2_10_pct": {**summarize(gap_values), **bootstrap_ci(gap_values, rng)},
    }
    write_csv(tables / "replay_bootstrap_ci.csv", [{"metric": k, **v} for k, v in replay_bootstrap.items()])
    corr = float(np.corrcoef(np.asarray(surrogate_values), np.asarray(replay_values))[0, 1]) if len(surrogate_values) > 1 else float("nan")

    sample_manifest = {
        "seed": int(args.seed),
        "forward": {"n": len(forward_sample_rows), "records": forward_sample_rows},
        "query": {
            "per_query_prediction_arrays_available": False,
            "n": 0,
            "metric_row_n": len(query_sample_rows),
            "records": query_sample_rows,
        },
        "candidate": {"n": len(candidate_sample_light), "records": candidate_sample_light},
        "replay": {
            "target_n": int(target_bands.shape[0]),
            "candidate_jsonl_n": len(replay_candidates),
            "candidate_level_replay_bands_available": False,
        },
    }
    write_json(run_dir / "sample_manifest.json", sample_manifest)

    metrics_summary = {
        "analysis_boundary": {
            "new_training_run": False,
            "new_mpb_meep_replay": False,
            "formal_replay_source": rel(paths["mixture_replay"], root),
            "headline_band_range": "bands_2_10",
        },
        "forward": {
            "selected_case_id": selected_case_id,
            "sample_n": len(forward_sample_rows),
            "sample_mre_bands_2_10_pct": summarize([row["mre_bands_2_10_pct"] for row in forward_sample_rows]),
            "top_case_avg_mre_bands_2_10_pct": forward_top[0]["avg_MRE_bands_2_10_pct"],
        },
        "query": {
            "per_query_prediction_arrays_available": False,
            "per_k_metric_sample_n": len(query_sample_rows),
            "protocol_summary": query_protocol_rows,
            "heldout_segment_gate_pass": boundary_map.get("heldout_segment_gate_pass"),
        },
        "preflight": preflight_metrics,
        "replay": {
            "metrics": replay_metrics,
            "target_n": int(target_bands.shape[0]),
            "replay_target_stats": summarize(replay_values),
            "surrogate_target_stats": summarize(surrogate_values),
            "gap_target_stats": summarize(gap_values),
            "surrogate_vs_replay_corr": corr,
        },
        "replay_bootstrap": replay_bootstrap,
        "baseline_comparison": baseline_rows,
    }
    write_json(run_dir / "metrics_summary.json", metrics_summary)
    write_json(run_dir / "results.json", {"status": "ok", "stage": EXPERIMENT_ID, "run_dir": rel(run_dir, root), "report_path": rel(report_path, root)})

    analysis_config = {
        "experiment_id": EXPERIMENT_ID,
        "seed": int(args.seed),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "inputs": {key: rel(value, root) for key, value in paths.items()},
        "report_path": rel(report_path, root),
    }
    write_json(run_dir / "analysis_config.json", analysis_config)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        git_commit = None
    write_json(
        run_dir / "run.meta.json",
        {
            "script": rel(Path(__file__), root),
            "experiment_id": EXPERIMENT_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "boundary": "offline analysis of existing artifacts; no new training; no new MPB/Meep replay",
        },
    )
    make_report(
        root=root,
        run_dir=run_dir,
        report_path=report_path,
        metrics=metrics_summary,
        paths=paths,
        samples=sample_manifest,
        artifact_index=artifact_index,
        baseline_rows=baseline_rows,
        forward_top=forward_top,
        query_protocol_rows=query_protocol_rows,
        replay_stats=metrics_summary["replay"],
        best_rows=best_rows,
        worst_rows=worst_rows,
    )
    log_lines.append(f"finished_at={datetime.now(timezone.utc).isoformat()}")
    log_lines.append(f"report_path={rel(report_path, root)}")
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": rel(run_dir, root), "report_path": rel(report_path, root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
