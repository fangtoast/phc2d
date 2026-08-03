"""Plot sampled inverse test625 target/surrogate/real-MPB band curves.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from phc2d.data import lift_rho_to_grid


PALETTE = {
    "target": "#24292F",
    "surrogate": "#C18F2E",
    "mpb": "#2364AA",
    "gap": "#B4433F",
    "grid": "#D8DEE4",
    "structure": "#2F73B7",
    "bg": "#DDF3DE",
}
DENOM_FLOOR = 1.0e-6
BINARY_CMAP = ListedColormap(["white", PALETTE["structure"]])


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


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def band_slice(nb: int, one_based_start: int = 2, one_based_end: int | None = None) -> slice:
    start = max(0, int(one_based_start) - 1)
    end = nb if one_based_end is None else min(nb, int(one_based_end))
    if start >= end:
        return slice(0, nb)
    return slice(start, end)


def mre_pct(target: np.ndarray, pred: np.ndarray, bands: slice) -> float:
    true = np.asarray(target[:, bands], dtype=float)
    value = np.asarray(pred[:, bands], dtype=float)
    denom = np.maximum(np.abs(true), DENOM_FLOOR)
    return float(100.0 * np.mean(np.abs(value - true) / denom))


def sample_positions(n_total: int, n_samples: int, seed: int) -> np.ndarray:
    if n_total <= 0:
        raise ValueError("No targets available for sampling.")
    n = min(max(1, int(n_samples)), int(n_total))
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_total, size=n, replace=False))


def selected_positions(
    per_target: Sequence[dict[str, str]],
    *,
    n_samples: int,
    seed: int,
    selection: str,
    metric_key: str,
) -> np.ndarray:
    normalized = selection.strip().lower()
    if normalized == "random":
        return sample_positions(len(per_target), n_samples, seed)
    scored: list[tuple[float, int]] = []
    for position, row in enumerate(per_target):
        value = to_float(row.get(metric_key))
        if value is not None:
            scored.append((value, position))
    if not scored:
        raise ValueError(f"No numeric values found for selection metric: {metric_key}")
    reverse = normalized == "worst"
    if normalized not in {"best", "worst"}:
        raise ValueError(f"Unknown selection mode: {selection}")
    scored.sort(key=lambda item: (item[0], item[1]), reverse=reverse)
    return np.asarray([position for _score, position in scored[: max(1, int(n_samples))]], dtype=int)


def infer_candidates_path(summary_run: Path, root: Path) -> Path | None:
    for name in ["source_shard_manifest.json", "shard_manifest.json"]:
        path = summary_run / name
        if not path.exists():
            continue
        payload = read_json(path)
        value = payload.get("candidates_path") if isinstance(payload, dict) else None
        if value:
            return resolve_path(root, value)
    return None


def infer_dataset_path(summary_run: Path, root: Path) -> Path | None:
    for name in ["source_shard_manifest.json", "shard_manifest.json"]:
        path = summary_run / name
        if not path.exists():
            continue
        payload = read_json(path)
        value = payload.get("dataset_path") if isinstance(payload, dict) else None
        if value:
            return resolve_path(root, value)
    return None


def load_candidate_records(candidates_path: Path | None) -> dict[int, dict[str, Any]]:
    if candidates_path is None or not candidates_path.exists():
        return {}
    payload = read_json(candidates_path)
    targets = payload.get("targets") if isinstance(payload, dict) else None
    if not isinstance(targets, list):
        return {}
    return {int(record["target_index"]): record for record in targets if isinstance(record, dict) and "target_index" in record}


def load_candidate_surrogate_predictions(
    *,
    candidates_path: Path | None,
    per_target: Sequence[dict[str, str]],
    target_shape: tuple[int, int],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if candidates_path is None or not candidates_path.exists():
        return None, {"available": False, "reason": "candidates_path unavailable"}
    by_index = load_candidate_records(candidates_path)
    if not by_index:
        return None, {"available": False, "reason": "candidates.json has no targets list"}
    arr = np.full((len(per_target), *target_shape), np.nan, dtype=np.float32)
    recovered = 0
    fallback = 0
    missing = 0
    for row_pos, row in enumerate(per_target):
        target_index = int(row["target_index"])
        record = by_index.get(target_index)
        if record is None:
            missing += 1
            continue
        selected = [int(item) for item in record.get("selected_indices", [])]
        predictions = record.get("selected_surrogate_predictions", [])
        best_candidate = row.get("best_candidate_selected_index")
        pred_values = None
        try:
            best_candidate_int = int(best_candidate) if best_candidate not in {None, ""} else None
        except ValueError:
            best_candidate_int = None
        if best_candidate_int is not None and best_candidate_int in selected:
            pred_pos = selected.index(best_candidate_int)
            if pred_pos < len(predictions):
                pred_values = predictions[pred_pos]
        if pred_values is None and record.get("best_surrogate_prediction") is not None:
            pred_values = record.get("best_surrogate_prediction")
            fallback += 1
        if pred_values is None:
            missing += 1
            continue
        pred = np.asarray(pred_values, dtype=np.float32).reshape(target_shape)
        arr[row_pos] = pred
        recovered += 1
    return arr, {
        "available": recovered > 0,
        "candidates_path": str(candidates_path),
        "target_count": len(per_target),
        "recovered_count": int(recovered),
        "fallback_best_surrogate_count": int(fallback),
        "missing_count": int(missing),
        "source": "selected_surrogate_predictions matched by best_candidate_selected_index",
    }


def choose_surrogate_array(
    *,
    summary_surrogate: np.ndarray,
    candidates_path: Path | None,
    per_target: Sequence[dict[str, str]],
) -> tuple[np.ndarray, dict[str, Any]]:
    finite_count = int(np.isfinite(summary_surrogate).sum())
    if finite_count > 0:
        return summary_surrogate, {
            "source": "summary_run/surrogate_pred_bands.npy",
            "finite_count": finite_count,
            "available": True,
        }
    recovered, info = load_candidate_surrogate_predictions(
        candidates_path=candidates_path,
        per_target=per_target,
        target_shape=tuple(int(v) for v in summary_surrogate.shape[1:]),
    )
    if recovered is not None and info.get("available"):
        return recovered, info
    return summary_surrogate, {"source": "summary_run/surrogate_pred_bands.npy", "finite_count": finite_count, **info}


def rho_to_grid(rho: np.ndarray) -> np.ndarray:
    arr = np.asarray(rho, dtype=np.float32)
    if arr.ndim == 2:
        return (arr >= 0.5).astype(float)
    flat = arr.reshape(-1)
    try:
        return (lift_rho_to_grid(flat) >= 0.5).astype(float)
    except Exception:
        side = int(np.sqrt(flat.size))
        if side * side != flat.size:
            raise ValueError(f"Cannot display rho with flattened size {flat.size}") from None
        return (flat.reshape(side, side) >= 0.5).astype(float)


def select_inverse_rho(record: dict[str, Any], best_candidate_index: int) -> np.ndarray:
    candidates = record.get("candidates", [])
    if isinstance(candidates, list) and 0 <= best_candidate_index < len(candidates):
        return np.asarray(candidates[best_candidate_index], dtype=np.float32)
    metadata = record.get("metadata", [])
    if isinstance(candidates, list) and isinstance(metadata, list):
        for pos, item in enumerate(metadata):
            if isinstance(item, dict) and item.get("candidate_index") == best_candidate_index and pos < len(candidates):
                return np.asarray(candidates[pos], dtype=np.float32)
    raw = record.get("raw_candidates", [])
    if isinstance(raw, list) and 0 <= best_candidate_index < len(raw):
        return np.asarray(raw[best_candidate_index], dtype=np.float32)
    raise ValueError(f"Cannot find inverse rho for best_candidate_index={best_candidate_index}")


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linewidth=0.6, color=PALETTE["grid"])
    ax.tick_params(labelsize=8)


def set_k_ticks(ax: plt.Axes, nk: int, *, compact: bool = False) -> None:
    if nk <= 1:
        ax.set_xticks([0])
        return
    ticks = [0, nk // 3, (2 * nk) // 3, nk - 1]
    labels = [r"$\Gamma$", "X", "M", r"$\Gamma$"]
    ax.set_xticks(ticks)
    ax.set_xticklabels([] if compact else labels)


def draw_structure(ax: plt.Axes, rho: np.ndarray, title: str) -> None:
    ax.imshow(rho_to_grid(rho), cmap=BINARY_CMAP, origin="lower", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title, fontsize=5.5, pad=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_color("#2F3A45")


def plot_target_replay_bands(ax: plt.Axes, target: np.ndarray, replay: np.ndarray) -> None:
    target = np.asarray(target, dtype=float)
    replay = np.asarray(replay, dtype=float)
    nk, nb = target.shape
    x = np.arange(nk)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, nb))
    for band in range(nb):
        ax.plot(x, target[:, band], color=colors[band], linewidth=0.48, alpha=0.86)
        ax.plot(x, replay[:, band], color=colors[band], linewidth=0.48, linestyle="--", alpha=0.72)
    set_k_ticks(ax, nk)
    ax.set_xlabel("k path", fontsize=5.2)
    ax.set_ylabel("frequency", fontsize=5.2)
    ax.grid(alpha=0.16, linewidth=0.35)
    ax.tick_params(labelsize=4.8, width=0.35, length=1.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)


def plot_structure_band_gallery(
    *,
    out_base: Path,
    positions: Sequence[int],
    per_target: Sequence[dict[str, str]],
    dataset_rho: np.ndarray,
    candidate_records: dict[int, dict[str, Any]],
    target_bands: np.ndarray,
    replay_bands: np.ndarray,
    cols: int = 4,
) -> list[Path]:
    if not candidate_records:
        raise ValueError("Candidate records are required for structure gallery.")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    n = len(positions)
    cols = min(max(1, int(cols)), max(1, n))
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(5.6 * cols, 1.85 * rows), facecolor=PALETTE["bg"])
    outer = fig.add_gridspec(rows, cols, left=0.012, right=0.995, top=0.975, bottom=0.028, wspace=0.08, hspace=0.34)
    for panel_index, position in enumerate(positions):
        row = per_target[int(position)]
        target_index = int(row["target_index"])
        best_candidate = int(row["best_candidate_selected_index"])
        record = candidate_records[target_index]
        sub = outer[panel_index // cols, panel_index % cols].subgridspec(
            1,
            2,
            width_ratios=[0.33, 1.46],
            wspace=0.10,
        )
        struct_grid = sub[0, 0].subgridspec(2, 1, hspace=0.22)
        ax_target = fig.add_subplot(struct_grid[0, 0])
        ax_inverse = fig.add_subplot(struct_grid[1, 0])
        ax_band = fig.add_subplot(sub[0, 1])

        draw_structure(ax_target, dataset_rho[target_index], "target structure")
        draw_structure(ax_inverse, select_inverse_rho(record, best_candidate), "inverse structure")
        plot_target_replay_bands(ax_band, target_bands[int(position)], replay_bands[int(position)])
        replay_b1 = mre_pct(target_bands[int(position)], replay_bands[int(position)], slice(0, 1))
        replay_b2 = mre_pct(target_bands[int(position)], replay_bands[int(position)], band_slice(target_bands.shape[-1], 2, target_bands.shape[-1]))
        ax_band.set_title(
            f"v2b current replay | target_index={target_index} | candidate={best_candidate} | MRE b1={replay_b1:.2f}%, b2-10={replay_b2:.2f}%",
            fontsize=6.0,
            pad=2.0,
        )
    handles = [
        plt.Line2D([0], [0], color="#30363D", linewidth=0.85, label="target bands"),
        plt.Line2D([0], [0], color="#30363D", linewidth=0.85, linestyle="--", label="MPB replay bands"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Inverse test625 sampled target/inverse structures and real-MPB band overlays", fontsize=10, x=0.012, ha="left")
    saved: list[Path] = []
    for suffix in [".png", ".svg"]:
        out = out_base.with_suffix(suffix)
        fig.savefig(out, dpi=210)
        saved.append(out)
    plt.close(fig)
    return saved


def plot_single_target(
    *,
    out_path: Path,
    position: int,
    row: dict[str, str],
    target: np.ndarray,
    surrogate: np.ndarray,
    replay: np.ndarray,
    bands: slice,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nk, nb = target.shape
    x = np.arange(nk)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), gridspec_kw={"width_ratios": [2.8, 1.0]})
    ax = axes[0]
    for band in range(nb):
        alpha = 0.38 if band == 0 else 0.72
        ax.plot(x, target[:, band], color=PALETTE["target"], linewidth=0.9, alpha=alpha, label="target" if band == 0 else None)
        ax.plot(
            x,
            surrogate[:, band],
            color=PALETTE["surrogate"],
            linewidth=0.85,
            linestyle="--",
            alpha=alpha,
            label="surrogate" if band == 0 else None,
        )
        ax.plot(
            x,
            replay[:, band],
            color=PALETTE["mpb"],
            linewidth=0.85,
            linestyle=":",
            alpha=alpha,
            label="real MPB replay" if band == 0 else None,
        )
    set_k_ticks(ax, nk)
    style_axes(ax)
    ax.set_xlabel("k path")
    ax.set_ylabel("normalized frequency")
    ax.set_title("Full-band curve overlay")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")

    ax_mean = axes[1]
    target_mean = np.mean(target[:, bands], axis=1)
    surrogate_mean = np.mean(surrogate[:, bands], axis=1)
    replay_mean = np.mean(replay[:, bands], axis=1)
    ax_mean.plot(x, target_mean, color=PALETTE["target"], linewidth=1.3, label="target")
    ax_mean.plot(x, surrogate_mean, color=PALETTE["surrogate"], linewidth=1.2, linestyle="--", label="surrogate")
    ax_mean.plot(x, replay_mean, color=PALETTE["mpb"], linewidth=1.2, linestyle=":", label="real MPB")
    set_k_ticks(ax_mean, nk)
    style_axes(ax_mean)
    ax_mean.set_title("Bands 2-end mean")
    ax_mean.set_xlabel("k path")
    ax_mean.tick_params(labelleft=False)

    replay_mre = mre_pct(target, replay, bands)
    surrogate_mre = mre_pct(target, surrogate, bands)
    gap = replay_mre - surrogate_mre
    target_index = row.get("target_index", str(position))
    fig.suptitle(
        f"target_index={target_index} | row={position} | replay={replay_mre:.2f}% | surrogate={surrogate_mre:.2f}% | gap={gap:.2f}%",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return {
        "position": int(position),
        "target_index": target_index,
        "figure": str(out_path),
        "best_of_k_replay_mre_bands_2_10_pct": row.get("best_of_k_replay_mre_bands_2_10_pct"),
        "best_of_k_surrogate_mre_bands_2_10_pct": row.get("best_of_k_surrogate_mre_bands_2_10_pct"),
        "surrogate_replay_gap_bands_2_10_pct": row.get("surrogate_replay_gap_bands_2_10_pct"),
        "recomputed_replay_mre_pct": replay_mre,
        "recomputed_surrogate_mre_pct": surrogate_mre,
        "recomputed_gap_pct": gap,
    }


def plot_overview(
    *,
    out_base: Path,
    positions: Sequence[int],
    rows: Sequence[dict[str, str]],
    target: np.ndarray,
    surrogate: np.ndarray,
    replay: np.ndarray,
    bands: slice,
) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    n = len(positions)
    cols = min(10, n)
    rows_count = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(2.25 * cols, 1.75 * rows_count), sharex=True)
    axes_arr = np.asarray(axes).reshape(rows_count, cols)
    for ax in axes_arr.flat:
        ax.set_axis_off()
    for panel_index, position in enumerate(positions):
        ax = axes_arr.flat[panel_index]
        ax.set_axis_on()
        x = np.arange(target.shape[1])
        ax.plot(x, np.mean(target[position, :, bands], axis=1), color=PALETTE["target"], linewidth=0.9)
        ax.plot(x, np.mean(surrogate[position, :, bands], axis=1), color=PALETTE["surrogate"], linewidth=0.85, linestyle="--")
        ax.plot(x, np.mean(replay[position, :, bands], axis=1), color=PALETTE["mpb"], linewidth=0.85, linestyle=":")
        set_k_ticks(ax, target.shape[1], compact=True)
        style_axes(ax)
        ax.tick_params(labelsize=5, labelleft=False)
        metric = to_float(rows[position].get("best_of_k_replay_mre_bands_2_10_pct")) if position < len(rows) else None
        label = rows[position].get("target_index", str(position)) if position < len(rows) else str(position)
        ax.set_title(f"{label} | {metric:.2f}%" if metric is not None else str(label), fontsize=6)
    handles = [
        plt.Line2D([0], [0], color=PALETTE["target"], linewidth=1.3, label="target"),
        plt.Line2D([0], [0], color=PALETTE["surrogate"], linewidth=1.3, linestyle="--", label="surrogate"),
        plt.Line2D([0], [0], color=PALETTE["mpb"], linewidth=1.3, linestyle=":", label="real MPB replay"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"Inverse test625 sampled band curves, n={n}, bands 2-end mean", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.975])
    saved: list[Path] = []
    for suffix in [".png", ".svg"]:
        out = out_base.with_suffix(suffix)
        fig.savefig(out, dpi=220)
        saved.append(out)
    plt.close(fig)
    return saved


def update_artifact_index(summary_run: Path, root: Path, generated: Sequence[Path]) -> None:
    index_path = summary_run / "artifact_index.json"
    payload = read_json(index_path) if index_path.exists() else {"artifacts": []}
    existing = {item.get("path") for item in payload.get("artifacts", []) if isinstance(item, dict)}
    for path in generated:
        item = {
            "path": rel(path, root),
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size) if path.exists() else None,
        }
        if item["path"] not in existing:
            payload.setdefault("artifacts", []).append(item)
            existing.add(item["path"])
    write_json(index_path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-run",
        type=Path,
        default=Path("runs/20260706/inverse_mpb_replay_mixture_v2_10000_test625_summary_20260706224424"),
        help="Summary run containing target_bands.npy, surrogate_pred_bands.npy, and mpb_replay_bands.npy.",
    )
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--output-subdir", default="band_curve_sample100")
    parser.add_argument("--output-prefix", default="band_curve_sample100")
    parser.add_argument("--individual-subdir", default="individual")
    parser.add_argument("--selection", choices=["random", "best", "worst"], default="random")
    parser.add_argument("--selection-metric", default="best_of_k_replay_mre_bands_2_10_pct")
    parser.add_argument("--band-start", type=int, default=2, help="One-based headline band start for mean panel.")
    parser.add_argument("--band-end", type=int, default=None, help="One-based inclusive headline band end.")
    parser.add_argument("--candidates", type=Path, help="Optional candidates.json for recovering selected surrogate curves.")
    parser.add_argument("--dataset", type=Path, help="Optional dataset npz for target rho structures.")
    parser.add_argument("--structure-cols", type=int, default=4, help="Columns in the structure+band gallery.")
    parser.add_argument("--skip-structure-gallery", action="store_true")
    return parser


def structure_gallery_suffix(output_prefix: str) -> str:
    prefix = str(output_prefix)
    return prefix[len("band_curve_") :] if prefix.startswith("band_curve_") else prefix


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    summary_run = resolve_path(root, args.summary_run)
    target = np.load(summary_run / "target_bands.npy")
    surrogate = np.load(summary_run / "surrogate_pred_bands.npy")
    replay = np.load(summary_run / "mpb_replay_bands.npy")
    if target.shape != surrogate.shape or target.shape != replay.shape:
        raise ValueError(f"Band array shapes differ: target={target.shape} surrogate={surrogate.shape} replay={replay.shape}")
    per_target = read_csv(summary_run / "summary_tables" / "per_target_replay_metrics.csv")
    if len(per_target) != target.shape[0]:
        raise ValueError(f"per_target rows {len(per_target)} do not match band arrays {target.shape[0]}")
    candidates_path = resolve_path(root, args.candidates) if args.candidates else infer_candidates_path(summary_run, root)
    dataset_path = resolve_path(root, args.dataset) if args.dataset else infer_dataset_path(summary_run, root)
    surrogate, surrogate_source = choose_surrogate_array(
        summary_surrogate=surrogate,
        candidates_path=candidates_path,
        per_target=per_target,
    )

    bands = band_slice(target.shape[-1], args.band_start, args.band_end)
    positions = selected_positions(
        per_target,
        n_samples=args.n_samples,
        seed=args.seed,
        selection=args.selection,
        metric_key=args.selection_metric,
    )
    output_dir = summary_run / "figures" / args.output_subdir
    individual_dir = output_dir / args.individual_subdir
    rows: list[dict[str, Any]] = []
    generated: list[Path] = []
    overview_paths = plot_overview(
        out_base=output_dir / f"{args.output_prefix}_overview",
        positions=positions,
        rows=per_target,
        target=target,
        surrogate=surrogate,
        replay=replay,
        bands=bands,
    )
    generated.extend(overview_paths)
    structure_gallery_paths: list[Path] = []
    structure_gallery_info: dict[str, Any] = {"available": False}
    if not args.skip_structure_gallery:
        candidate_records = load_candidate_records(candidates_path)
        if dataset_path is not None and dataset_path.exists() and candidate_records:
            with np.load(dataset_path) as dataset:
                if "rho" not in dataset:
                    raise ValueError(f"Dataset has no rho array: {dataset_path}")
                structure_gallery_paths = plot_structure_band_gallery(
                    out_base=output_dir / f"structure_band_gallery_{structure_gallery_suffix(args.output_prefix)}",
                    positions=positions,
                    per_target=per_target,
                    dataset_rho=np.asarray(dataset["rho"]),
                    candidate_records=candidate_records,
                    target_bands=target,
                    replay_bands=replay,
                    cols=int(args.structure_cols),
                )
            generated.extend(structure_gallery_paths)
            structure_gallery_info = {
                "available": True,
                "dataset_path": rel(dataset_path, root),
                "candidates_path": rel(candidates_path, root) if candidates_path else None,
                "figures": [rel(path, root) for path in structure_gallery_paths],
                "layout": {"columns": int(args.structure_cols), "samples": int(len(positions))},
            }
        else:
            structure_gallery_info = {
                "available": False,
                "dataset_path": rel(dataset_path, root) if dataset_path else None,
                "candidates_path": rel(candidates_path, root) if candidates_path else None,
                "reason": "dataset_path or candidate records unavailable",
            }
    for order, position in enumerate(positions, start=1):
        row = per_target[int(position)]
        target_index = row.get("target_index", f"pos{position}")
        out_path = individual_dir / f"{order:03d}_target_{target_index}_band_curve.png"
        sample_row = plot_single_target(
            out_path=out_path,
            position=int(position),
            row=row,
            target=target[int(position)],
            surrogate=surrogate[int(position)],
            replay=replay[int(position)],
            bands=bands,
        )
        sample_row["figure"] = rel(out_path, root)
        rows.append(sample_row)
        generated.append(out_path)

    sample_index = output_dir / f"{args.output_prefix}_index.csv"
    write_csv(sample_index, rows)
    generated.append(sample_index)
    manifest = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary_run": rel(summary_run, root),
        "n_samples_requested": int(args.n_samples),
        "n_samples_written": int(len(rows)),
        "seed": int(args.seed),
        "selection": str(args.selection),
        "selection_metric": str(args.selection_metric),
        "sampling": "uniform_without_replacement_sorted_by_position" if args.selection == "random" else "metric_sorted",
        "band_start_one_based": int(args.band_start),
        "band_end_one_based": int(args.band_end) if args.band_end is not None else int(target.shape[-1]),
        "overview_figures": [rel(path, root) for path in overview_paths],
        "sample_index": rel(sample_index, root),
        "individual_dir": rel(individual_dir, root),
        "surrogate_curve_source": surrogate_source,
        "structure_band_gallery": structure_gallery_info,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    generated.append(manifest_path)
    update_artifact_index(summary_run, root, generated)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
