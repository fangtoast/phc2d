"""Plot A2 objective-bridge inverse structures and real-MPB band overlays.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from phc2d.data import lift_rho_to_grid


PALETTE = {
    "blue": "#3775BA",
    "red": "#B64342",
    "green_bg": "#DDF3DE",
    "grid": "#CFCECE",
    "text": "#272727",
}
STRUCTURE_CMAP = ListedColormap(["white", PALETTE["blue"]])
CHANGE_CMAP = ListedColormap(["white", PALETTE["red"]])
METHOD_LABELS = {
    "single_mode_generator": "Objective-bridge single-mode",
    "mixture_generator": "Objective-bridge mixture",
}
SINGLE_METHOD = "single_mode_generator"
MIXTURE_METHOD = "mixture_generator"
PAIRED_SELECTION_ORDER = (
    "joint_best",
    "joint_median",
    "largest_disagreement",
    "joint_worst",
    "strongest_single_win",
    "strongest_mixture_win",
    "largest_calibration_gap",
    "random",
)
DENOM_FLOOR = 1.0e-6


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    header = json.dumps(
        {"shape": list(arr.shape), "dtype": str(arr.dtype)},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + arr.tobytes()).hexdigest()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 1.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def rho_to_grid(rho: np.ndarray) -> np.ndarray:
    arr = np.asarray(rho, dtype=np.float32)
    if arr.ndim == 2:
        return (arr >= 0.5).astype(float)
    return (lift_rho_to_grid(arr.reshape(-1)) >= 0.5).astype(float)


def mre_pct(target: np.ndarray, pred: np.ndarray, start_band_zero_based: int = 1) -> float:
    true = np.asarray(target, dtype=float)[..., start_band_zero_based:]
    value = np.asarray(pred, dtype=float)[..., start_band_zero_based:]
    denom = np.maximum(np.abs(true), DENOM_FLOOR)
    return float(100.0 * np.mean(np.abs(value - true) / denom))


def choose_best_replays(run_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = read_csv(run_dir / "replay" / "replay_candidates.csv")
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row["method"] not in METHOD_LABELS or row.get("status") not in {"computed", "cached", "reused"}:
            continue
        key = (row["method"], int(row["target_index"]))
        score = float(row["mpb_replay_mre"])
        current = best.get(key)
        if current is None or score < float(current["mpb_replay_mre"]):
            best[key] = {
                **row,
                "target_index": int(row["target_index"]),
                "candidate_index": int(row["candidate_index"]),
                "rank_position": int(row["rank_position"]),
                "mpb_replay_mre": score,
            }
    return best


def load_records(run_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    manifest = read_json(run_dir / "candidates" / "candidate_manifest.frozen.json")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("candidate_manifest.frozen.json has no records list")
    return {
        (str(record["method"]), int(record["target_index"])): record
        for record in records
        if record.get("method") in METHOD_LABELS
    }


def select_positions(cases: Sequence[dict[str, Any]], seed: int) -> dict[str, int]:
    if len(cases) < 4:
        raise ValueError("At least four cases are required for best/median/worst/random selection.")
    order = np.argsort([float(case["replay_mre_pct"]) for case in cases])
    rng = np.random.default_rng(seed)
    fixed = {int(order[0]), int(order[len(order) // 2]), int(order[-1])}
    random_candidates = [index for index in range(len(cases)) if index not in fixed]
    random_position = int(rng.choice(random_candidates))
    return {
        "best": int(order[0]),
        "median": int(order[len(order) // 2]),
        "worst": int(order[-1]),
        "random": random_position,
    }


def case_map(method_cases: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(case["target_index"]): case for case in method_cases}


def paired_case_rows(cases: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    single = case_map(cases[SINGLE_METHOD])
    mixture = case_map(cases[MIXTURE_METHOD])
    if set(single) != set(mixture):
        missing_single = sorted(set(mixture).difference(single))
        missing_mixture = sorted(set(single).difference(mixture))
        raise ValueError(
            "Paired method target sets differ: "
            f"missing_single={missing_single[:8]}, missing_mixture={missing_mixture[:8]}"
        )
    rows: list[dict[str, Any]] = []
    for target_index in sorted(single):
        single_case = single[target_index]
        mixture_case = mixture[target_index]
        single_replay = float(single_case["replay_mre_pct"])
        mixture_replay = float(mixture_case["replay_mre_pct"])
        single_gap = abs(single_replay - float(single_case["surrogate_mre_pct"]))
        mixture_gap = abs(mixture_replay - float(mixture_case["surrogate_mre_pct"]))
        rows.append(
            {
                "target_index": target_index,
                "single_case": single_case,
                "mixture_case": mixture_case,
                "joint_score": 0.5 * (single_replay + mixture_replay),
                "disagreement": abs(single_replay - mixture_replay),
                "single_advantage": mixture_replay - single_replay,
                "mixture_advantage": single_replay - mixture_replay,
                "calibration_gap": max(single_gap, mixture_gap),
            }
        )
    return rows


def _first_unused(
    ranked_rows: Sequence[dict[str, Any]],
    used: set[int],
) -> dict[str, Any]:
    for row in ranked_rows:
        target_index = int(row["target_index"])
        if target_index not in used:
            used.add(target_index)
            return row
    raise ValueError("No unused paired target remains for deterministic selection.")


def select_paired_targets(
    cases: dict[str, list[dict[str, Any]]],
    seed: int,
) -> dict[str, dict[str, Any]]:
    rows = paired_case_rows(cases)
    if len(rows) < len(PAIRED_SELECTION_ORDER):
        raise ValueError(
            f"At least {len(PAIRED_SELECTION_ORDER)} paired targets are required; received {len(rows)}."
        )
    joint_values = np.asarray([float(row["joint_score"]) for row in rows], dtype=float)
    joint_median = float(np.median(joint_values))
    rankings = {
        "joint_best": sorted(rows, key=lambda row: (float(row["joint_score"]), int(row["target_index"]))),
        "joint_median": sorted(
            rows,
            key=lambda row: (abs(float(row["joint_score"]) - joint_median), int(row["target_index"])),
        ),
        "largest_disagreement": sorted(
            rows,
            key=lambda row: (-float(row["disagreement"]), int(row["target_index"])),
        ),
        "joint_worst": sorted(
            rows,
            key=lambda row: (-float(row["joint_score"]), int(row["target_index"])),
        ),
        "strongest_single_win": sorted(
            rows,
            key=lambda row: (-float(row["single_advantage"]), int(row["target_index"])),
        ),
        "strongest_mixture_win": sorted(
            rows,
            key=lambda row: (-float(row["mixture_advantage"]), int(row["target_index"])),
        ),
        "largest_calibration_gap": sorted(
            rows,
            key=lambda row: (-float(row["calibration_gap"]), int(row["target_index"])),
        ),
    }
    used: set[int] = set()
    selected: dict[str, dict[str, Any]] = {}
    for category in PAIRED_SELECTION_ORDER[:-1]:
        selected[category] = _first_unused(rankings[category], used)
    rng = np.random.default_rng(seed)
    random_rows = [rows[int(index)] for index in rng.permutation(len(rows))]
    selected["random"] = _first_unused(random_rows, used)
    return selected


def select_method_extremes(
    method_cases: Sequence[dict[str, Any]],
    count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n = max(1, int(count))
    if len(method_cases) < 2 * n:
        raise ValueError(f"Need at least {2 * n} method cases for disjoint best/worst sets.")
    ordered = sorted(
        method_cases,
        key=lambda case: (float(case["replay_mre_pct"]), int(case["target_index"])),
    )
    return ordered[:n], list(reversed(ordered[-n:]))


def load_cases(
    run_dir: Path,
    dataset_path: Path,
    expected_target_count: int,
) -> dict[str, list[dict[str, Any]]]:
    records = load_records(run_dir)
    best_replays = choose_best_replays(run_dir)
    dataset = np.load(dataset_path, allow_pickle=False)
    dataset_rho = np.asarray(dataset["rho"])
    cases: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_LABELS}
    for (method, target_index), replay in sorted(best_replays.items()):
        record = records[(method, target_index)]
        candidate_index = int(replay["candidate_index"])
        candidates = record["candidates"]
        surrogate_predictions = record["surrogate_predictions"]
        if not (0 <= candidate_index < len(candidates)):
            raise ValueError(f"Candidate index {candidate_index} is out of range for {method}, target {target_index}")
        target_bands = np.asarray(record["condition"]["Wh"], dtype=np.float32)
        surrogate_bands = np.asarray(surrogate_predictions[candidate_index], dtype=np.float32)
        replay_bands = np.load(
            run_dir / "replay" / "structure_cache" / f"{replay['structure_sha256']}.npy",
            allow_pickle=False,
        )
        replay_mre = mre_pct(target_bands, replay_bands)
        recorded_replay_mre = 100.0 * float(replay["mpb_replay_mre"])
        if not np.isclose(replay_mre, recorded_replay_mre, rtol=1.0e-6, atol=1.0e-5):
            raise ValueError(
                "Recomputed replay MRE does not match replay_candidates.csv for "
                f"{method}, target {target_index}, candidate {candidate_index}: "
                f"{replay_mre:.10f}% != {recorded_replay_mre:.10f}%"
            )
        candidate_vector = np.asarray(candidates[candidate_index], dtype=np.float32).reshape(-1)
        candidate_sha256 = hash_array(candidate_vector)
        if candidate_sha256 != replay["structure_sha256"]:
            raise ValueError(
                "Candidate structure SHA-256 does not match replay cache key for "
                f"{method}, target {target_index}, candidate {candidate_index}: "
                f"{candidate_sha256} != {replay['structure_sha256']}"
            )
        surrogate_mre = mre_pct(target_bands, surrogate_bands)
        cases[method].append(
            {
                "method": method,
                "target_index": target_index,
                "candidate_index": candidate_index,
                "rank_position": int(replay["rank_position"]),
                "structure_sha256": replay["structure_sha256"],
                "target_rho": np.asarray(dataset_rho[target_index]),
                "candidate_rho": np.asarray(candidates[candidate_index]),
                "target_bands": target_bands,
                "surrogate_bands": surrogate_bands,
                "replay_bands": replay_bands,
                "replay_mre_pct": replay_mre,
                "recorded_replay_mre_pct": recorded_replay_mre,
                "surrogate_mre_pct": surrogate_mre,
                "gap_pct": replay_mre - surrogate_mre,
            }
        )
    dataset.close()
    for method, method_cases in cases.items():
        if len(method_cases) != expected_target_count:
            raise ValueError(f"Expected {expected_target_count} cases for {method}, received {len(method_cases)}")
    return cases


def draw_structure(ax: plt.Axes, rho: np.ndarray, title: str, *, changed: bool = False) -> None:
    ax.imshow(
        rho,
        cmap=CHANGE_CMAP if changed else STRUCTURE_CMAP,
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("#4D4D4D")


def draw_bands(
    ax: plt.Axes,
    target: np.ndarray,
    surrogate: np.ndarray,
    replay: np.ndarray,
    *,
    compact: bool,
    y_limits: tuple[float, float] | None = None,
    show_ylabel: bool = True,
) -> None:
    target = np.asarray(target, dtype=float)
    surrogate = np.asarray(surrogate, dtype=float)
    replay = np.asarray(replay, dtype=float)
    if target.shape != surrogate.shape or target.shape != replay.shape:
        raise ValueError(
            f"Band shapes must match: target={target.shape}, surrogate={surrogate.shape}, replay={replay.shape}"
        )
    nk, nb = target.shape
    x = np.arange(nk)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, nb))
    for band in range(nb):
        ax.plot(x, target[:, band], color=colors[band], linewidth=0.95 if not compact else 0.62, alpha=0.94)
        ax.plot(
            x,
            surrogate[:, band],
            color=colors[band],
            linewidth=0.9 if not compact else 0.58,
            linestyle=":",
            alpha=0.84,
        )
        ax.plot(
            x,
            replay[:, band],
            color=colors[band],
            linewidth=0.95 if not compact else 0.62,
            linestyle="--",
            alpha=0.82,
        )
    ticks = [0, nk // 3, (2 * nk) // 3, nk - 1]
    ax.set_xticks(ticks, [r"$\Gamma$", "X", "M", r"$\Gamma$"])
    ax.grid(color=PALETTE["grid"], alpha=0.28, linewidth=0.5)
    ax.tick_params(labelsize=6 if compact else 8, width=0.7, length=2.5)
    ax.set_xlabel("k path", fontsize=7 if compact else 9)
    if not compact and show_ylabel:
        ax.set_ylabel(r"$\omega a/(2\pi c)$", fontsize=9)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#4D4D4D")


def save_figure(fig: plt.Figure, output_base: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in formats:
        path = output_base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_four_case_figure(
    method: str,
    method_cases: Sequence[dict[str, Any]],
    selected: dict[str, int],
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    labels = [("best", "a"), ("median", "b"), ("worst", "c"), ("random", "d")]
    fig = plt.figure(figsize=(12.0, 8.7), facecolor="white")
    outer = fig.add_gridspec(2, 2, left=0.035, right=0.985, top=0.92, bottom=0.07, wspace=0.22, hspace=0.34)
    for panel, (selection_name, letter) in enumerate(labels):
        case = method_cases[selected[selection_name]]
        sub = outer[panel // 2, panel % 2].subgridspec(2, 2, width_ratios=[0.72, 2.7], hspace=0.18, wspace=0.13)
        ax_target = fig.add_subplot(sub[0, 0])
        ax_candidate = fig.add_subplot(sub[1, 0])
        ax_band = fig.add_subplot(sub[:, 1])
        draw_structure(ax_target, rho_to_grid(case["target_rho"]), "Target")
        draw_structure(ax_candidate, rho_to_grid(case["candidate_rho"]), "Inverse candidate")
        draw_bands(
            ax_band,
            case["target_bands"],
            case["surrogate_bands"],
            case["replay_bands"],
            compact=False,
        )
        ax_band.set_title(
            (
                f"({letter}) {selection_name.capitalize()} sample | target {case['target_index']} | "
                f"MPB {case['replay_mre_pct']:.2f}% | surrogate {case['surrogate_mre_pct']:.2f}% | "
                f"gap {case['gap_pct']:+.2f} pp"
            ),
            fontsize=9.2,
            pad=5,
        )
    handles = [
        plt.Line2D([0], [0], color="#30363D", linewidth=1.2, label="Target bands"),
        plt.Line2D([0], [0], color="#30363D", linewidth=1.2, linestyle=":", label="Surrogate bands"),
        plt.Line2D([0], [0], color="#30363D", linewidth=1.2, linestyle="--", label="Real-MPB replay bands"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(
        f"{METHOD_LABELS[method]}: representative inverse-design results on test625",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    return save_figure(fig, output_base, formats, dpi=300)


def plot_gallery(
    method: str,
    selected_cases: Sequence[dict[str, Any]],
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    cols = 4
    rows = int(np.ceil(len(selected_cases) / cols))
    fig = plt.figure(figsize=(15.5, 3.15 * rows), facecolor=PALETTE["green_bg"])
    outer = fig.add_gridspec(
        rows,
        cols,
        left=0.018,
        right=0.995,
        top=0.965,
        bottom=0.025,
        wspace=0.14,
        hspace=0.30,
    )
    for panel, case in enumerate(selected_cases):
        sub = outer[panel // cols, panel % cols].subgridspec(2, 2, width_ratios=[0.58, 2.75], hspace=0.12, wspace=0.10)
        ax_target = fig.add_subplot(sub[0, 0])
        ax_candidate = fig.add_subplot(sub[1, 0])
        ax_band = fig.add_subplot(sub[:, 1])
        draw_structure(ax_target, rho_to_grid(case["target_rho"]), "target")
        draw_structure(ax_candidate, rho_to_grid(case["candidate_rho"]), "inverse")
        draw_bands(
            ax_band,
            case["target_bands"],
            case["surrogate_bands"],
            case["replay_bands"],
            compact=True,
        )
        ax_band.set_title(
            (
                f"target={case['target_index']} | cand={case['candidate_index']} | "
                f"MPB={case['replay_mre_pct']:.2f}% | surr={case['surrogate_mre_pct']:.2f}%"
            ),
            fontsize=6.5,
            pad=2,
        )
    fig.suptitle(
        f"{METHOD_LABELS[method]} | fixed-seed random sample of {len(selected_cases)} test targets",
        fontsize=12,
        fontweight="bold",
        x=0.018,
        ha="left",
    )
    return save_figure(fig, output_base, formats, dpi=260)


def paired_y_limits(row: dict[str, Any]) -> tuple[float, float]:
    arrays = []
    for method_key in ("single_case", "mixture_case"):
        case = row[method_key]
        arrays.extend([case["target_bands"], case["surrogate_bands"], case["replay_bands"]])
    values = np.concatenate([np.asarray(value, dtype=float).reshape(-1) for value in arrays])
    low = float(np.min(values))
    high = float(np.max(values))
    pad = max(0.03, 0.04 * (high - low))
    return low - pad, high + pad


def plot_paired_grid(
    selected: dict[str, dict[str, Any]],
    categories: Sequence[str],
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    category_labels = {
        "joint_best": "Joint best",
        "joint_median": "Joint median",
        "largest_disagreement": "Largest disagreement",
        "joint_worst": "Joint worst",
        "strongest_single_win": "Strongest single-mode win",
        "strongest_mixture_win": "Strongest mixture win",
        "largest_calibration_gap": "Largest calibration gap",
        "random": "Fixed random",
    }
    rows = len(categories)
    fig = plt.figure(figsize=(17.2, 2.85 * rows + 1.15), facecolor="white")
    outer = fig.add_gridspec(
        rows,
        5,
        width_ratios=[0.8, 0.8, 2.45, 0.8, 2.45],
        left=0.035,
        right=0.992,
        top=0.93,
        bottom=0.055,
        wspace=0.17,
        hspace=0.44,
    )
    for row_position, category in enumerate(categories):
        row = selected[category]
        single = row["single_case"]
        mixture = row["mixture_case"]
        y_limits = paired_y_limits(row)
        axes = [fig.add_subplot(outer[row_position, col]) for col in range(5)]
        draw_structure(
            axes[0],
            rho_to_grid(single["target_rho"]),
            f"{category_labels[category]}\nTarget {row['target_index']}",
        )
        draw_structure(
            axes[1],
            rho_to_grid(single["candidate_rho"]),
            f"Single candidate {single['candidate_index']}",
        )
        draw_bands(
            axes[2],
            single["target_bands"],
            single["surrogate_bands"],
            single["replay_bands"],
            compact=True,
            y_limits=y_limits,
        )
        axes[2].set_title(
            (
                f"Single-mode | MPB {single['replay_mre_pct']:.2f}% | "
                f"surr {single['surrogate_mre_pct']:.2f}% | gap {single['gap_pct']:+.2f} pp"
            ),
            fontsize=7.4,
            pad=3,
        )
        draw_structure(
            axes[3],
            rho_to_grid(mixture["candidate_rho"]),
            f"Mixture candidate {mixture['candidate_index']}",
        )
        draw_bands(
            axes[4],
            mixture["target_bands"],
            mixture["surrogate_bands"],
            mixture["replay_bands"],
            compact=True,
            y_limits=y_limits,
            show_ylabel=False,
        )
        axes[4].set_title(
            (
                f"Mixture | MPB {mixture['replay_mre_pct']:.2f}% | "
                f"surr {mixture['surrogate_mre_pct']:.2f}% | gap {mixture['gap_pct']:+.2f} pp"
            ),
            fontsize=7.4,
            pad=3,
        )
    handles = [
        plt.Line2D([0], [0], color="#30363D", linewidth=1.2, label="Target"),
        plt.Line2D([0], [0], color="#30363D", linewidth=1.2, linestyle=":", label="Surrogate"),
        plt.Line2D([0], [0], color="#30363D", linewidth=1.2, linestyle="--", label="Real MPB"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(
        "A2 inverse comparison on identical test targets",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    return save_figure(fig, output_base, formats, dpi=300)


def plot_best_worst_gallery(
    method: str,
    best_cases: Sequence[dict[str, Any]],
    worst_cases: Sequence[dict[str, Any]],
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    selected_cases = list(best_cases) + list(worst_cases)
    cols = 4
    rows = int(np.ceil(len(selected_cases) / cols))
    best_rows = int(np.ceil(len(best_cases) / cols))
    fig = plt.figure(figsize=(16.0, 3.15 * rows + 0.7), facecolor="white")
    outer = fig.add_gridspec(
        rows,
        cols,
        left=0.02,
        right=0.995,
        top=0.94,
        bottom=0.025,
        wspace=0.14,
        hspace=0.34,
    )
    for row_position in range(rows):
        background = "#EEF7EE" if row_position < best_rows else "#FBEFEF"
        bbox = outer[row_position, :].get_position(fig)
        fig.add_artist(
            Rectangle(
                (bbox.x0 - 0.008, bbox.y0 - 0.01),
                bbox.width + 0.016,
                bbox.height + 0.02,
                transform=fig.transFigure,
                facecolor=background,
                edgecolor="none",
                zorder=-10,
            )
        )
    for panel, case in enumerate(selected_cases):
        group = "Best" if panel < len(best_cases) else "Worst"
        sub = outer[panel // cols, panel % cols].subgridspec(
            2,
            2,
            width_ratios=[0.58, 2.75],
            hspace=0.12,
            wspace=0.10,
        )
        ax_target = fig.add_subplot(sub[0, 0])
        ax_candidate = fig.add_subplot(sub[1, 0])
        ax_band = fig.add_subplot(sub[:, 1])
        draw_structure(ax_target, rho_to_grid(case["target_rho"]), "target")
        draw_structure(ax_candidate, rho_to_grid(case["candidate_rho"]), "inverse")
        draw_bands(
            ax_band,
            case["target_bands"],
            case["surrogate_bands"],
            case["replay_bands"],
            compact=True,
        )
        ax_band.set_title(
            (
                f"{group} | target={case['target_index']} | MPB={case['replay_mre_pct']:.2f}% | "
                f"surr={case['surrogate_mre_pct']:.2f}%"
            ),
            fontsize=6.5,
            pad=2,
        )
    fig.suptitle(
        f"{METHOD_LABELS[method]} | best {len(best_cases)} and worst {len(worst_cases)} by real-MPB MRE",
        fontsize=12.5,
        fontweight="bold",
    )
    return save_figure(fig, output_base, formats, dpi=300)


def plot_individual_page(
    target_index: int,
    single: dict[str, Any],
    mixture: dict[str, Any],
    label: str,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    target = rho_to_grid(single["target_rho"])
    single_candidate = rho_to_grid(single["candidate_rho"])
    mixture_candidate = rho_to_grid(mixture["candidate_rho"])
    single_changed = np.not_equal(target, single_candidate).astype(float)
    mixture_changed = np.not_equal(target, mixture_candidate).astype(float)
    row = {"single_case": single, "mixture_case": mixture}
    y_limits = paired_y_limits(row)
    fig = plt.figure(figsize=(16.0, 8.0), facecolor="white")
    outer = fig.add_gridspec(
        2,
        5,
        height_ratios=[1.05, 1.2],
        left=0.04,
        right=0.985,
        top=0.91,
        bottom=0.10,
        hspace=0.32,
        wspace=0.22,
    )
    structure_axes = [fig.add_subplot(outer[0, col]) for col in range(5)]
    draw_structure(structure_axes[0], target, "Target")
    draw_structure(structure_axes[1], single_candidate, "Single candidate")
    draw_structure(structure_axes[2], single_changed, "Single changed pixels", changed=True)
    draw_structure(structure_axes[3], mixture_candidate, "Mixture candidate")
    draw_structure(structure_axes[4], mixture_changed, "Mixture changed pixels", changed=True)
    ax_single = fig.add_subplot(outer[1, 0:2])
    ax_mixture = fig.add_subplot(outer[1, 3:5])
    draw_bands(
        ax_single,
        single["target_bands"],
        single["surrogate_bands"],
        single["replay_bands"],
        compact=False,
        y_limits=y_limits,
    )
    draw_bands(
        ax_mixture,
        mixture["target_bands"],
        mixture["surrogate_bands"],
        mixture["replay_bands"],
        compact=False,
        y_limits=y_limits,
    )
    ax_single.set_title(
        (
            f"Single-mode | candidate {single['candidate_index']} | MPB {single['replay_mre_pct']:.3f}% | "
            f"surrogate {single['surrogate_mre_pct']:.3f}% | gap {single['gap_pct']:+.3f} pp"
        ),
        fontsize=9.5,
    )
    ax_mixture.set_title(
        (
            f"Mixture | candidate {mixture['candidate_index']} | MPB {mixture['replay_mre_pct']:.3f}% | "
            f"surrogate {mixture['surrogate_mre_pct']:.3f}% | gap {mixture['gap_pct']:+.3f} pp"
        ),
        fontsize=9.5,
    )
    handles = [
        plt.Line2D([0], [0], color="#30363D", linewidth=1.2, label="Target"),
        plt.Line2D([0], [0], color="#30363D", linewidth=1.2, linestyle=":", label="Surrogate"),
        plt.Line2D([0], [0], color="#30363D", linewidth=1.2, linestyle="--", label="Real MPB"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle(f"Target {target_index} | {label}", fontsize=15, fontweight="bold", y=0.975)
    fig.text(
        0.5,
        0.012,
        "Changed pixels are descriptive only: inverse design is a multi-solution problem, so pixel difference is not a quality metric.",
        ha="center",
        fontsize=9.5,
        color=PALETTE["text"],
    )
    return save_figure(fig, output_base, formats, dpi=300)


def case_manifest_row(case: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        **extra,
        "method": case["method"],
        "target_index": int(case["target_index"]),
        "candidate_index": int(case["candidate_index"]),
        "rank_position": int(case["rank_position"]),
        "structure_sha256": str(case["structure_sha256"]),
        "real_mpb_replay_mre_bands_2_10_pct": float(case["replay_mre_pct"]),
        "surrogate_mre_bands_2_10_pct": float(case["surrogate_mre_pct"]),
        "surrogate_replay_gap_pp": float(case["gap_pct"]),
    }


def build_captions_markdown(
    paired_selected: dict[str, dict[str, Any]],
    random_count: int,
    extreme_count: int,
    seed: int,
) -> str:
    main_targets = ", ".join(
        str(paired_selected[category]["target_index"])
        for category in PAIRED_SELECTION_ORDER[:4]
    )
    supplement_targets = ", ".join(
        str(paired_selected[category]["target_index"])
        for category in PAIRED_SELECTION_ORDER
    )
    return f"""<!--
Purpose: Captions for A2 objective-bridge inverse visualization package.
Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
-->

# A2 inverse visualization captions

All examples are selected post hoc from the frozen 625-target real-MPB replay and are used only for visualization. They do not affect training, checkpoint selection, candidate generation, or ranking. Reported errors are bands 2–10 mean relative errors.

## Main paired figure

Four identical test targets ({main_targets}) compare single-mode and mixture inverse candidates. The rows are joint best, joint median, largest method disagreement, and joint worst according to the mean of the two real-MPB errors. Solid, dotted, and dashed curves denote target, surrogate, and real-MPB bands, respectively.

## Paired supplement

Eight distinct paired targets ({supplement_targets}) extend the main figure with the strongest single-mode win, strongest mixture win, largest calibration gap, and a fixed-seed random target. Duplicate categories are resolved by choosing the next unused target under the relevant ranking.

## Method-level representative figures

Each 2-by-2 figure shows the method-specific best, median, worst, and fixed-seed random target according to real-MPB error.

## Best/worst galleries

Each method gallery shows the best {extreme_count} and worst {extreme_count} targets in strict real-MPB error order. Background shading separates the descriptive groups and does not indicate statistical significance.

## Individual pages

Each page compares the target, both inverse candidates, changed-pixel masks, and calibrated band overlays. Changed pixels are descriptive only: inverse design is a multi-solution problem, so pixel difference is not a quality metric.

## Random gallery

The two random galleries use the same {random_count} target indices sampled with seed {seed}. The shared target list is stored in `random40_target_list.csv`.
"""


def artifact_rows(output_dir: Path, paths: Sequence[Path]) -> list[dict[str, Any]]:
    unique_paths = sorted({path.resolve() for path in paths if path.exists()})
    return [
        {
            "path": path.relative_to(output_dir.resolve()).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in unique_paths
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/20260728/a2_inverse_objective_bridge_seed0_test625_v1_202607280842"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/query_dataset_v2_pilot_10000_mpb.npz"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--gallery-count", type=int, default=40)
    parser.add_argument("--extreme-count", type=int, default=12)
    parser.add_argument("--expected-target-count", type=int, default=625)
    parser.add_argument(
        "--figure-set",
        choices=["all", "paired", "representative", "best-worst", "individual", "random40"],
        default="all",
    )
    parser.add_argument("--formats", nargs="+", choices=["png", "svg", "pdf"], default=["png", "svg", "pdf"])
    args = parser.parse_args(argv)

    root = repo_root()
    run_dir = resolve_path(root, args.run_dir)
    dataset_path = resolve_path(root, args.dataset)
    output_dir = (
        resolve_path(root, args.output_dir)
        if args.output_dir is not None
        else run_dir / "figures" / "inverse_examples_v2_paired"
    )
    required = [
        run_dir / "candidates" / "candidate_manifest.frozen.json",
        run_dir / "replay" / "replay_candidates.csv",
        dataset_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    apply_style()
    cases = load_cases(run_dir, dataset_path, int(args.expected_target_count))
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    selection_rows: list[dict[str, Any]] = []
    paired_selected = select_paired_targets(cases, int(args.seed))
    representative_selections = {
        method: select_positions(method_cases, int(args.seed) + method_index)
        for method_index, (method, method_cases) in enumerate(cases.items())
    }
    extreme_selections = {
        method: select_method_extremes(method_cases, int(args.extreme_count))
        for method, method_cases in cases.items()
    }
    paired_rows = paired_case_rows(cases)
    paired_by_target = {int(row["target_index"]): row for row in paired_rows}
    target_indices = np.asarray(sorted(paired_by_target), dtype=int)
    gallery_count = min(max(1, int(args.gallery_count)), len(target_indices))
    random_rng = np.random.default_rng(int(args.seed))
    random_target_indices = sorted(
        int(value)
        for value in random_rng.choice(target_indices, size=gallery_count, replace=False)
    )
    individual_labels: dict[int, set[str]] = {}
    for category, row in paired_selected.items():
        individual_labels.setdefault(int(row["target_index"]), set()).add(category)
    for method in METHOD_LABELS:
        best_cases, worst_cases = extreme_selections[method]
        individual_labels.setdefault(int(best_cases[0]["target_index"]), set()).add(
            f"{method}_best"
        )
        individual_labels.setdefault(int(worst_cases[0]["target_index"]), set()).add(
            f"{method}_worst"
        )

    for category in PAIRED_SELECTION_ORDER:
        row = paired_selected[category]
        for method_key, method in (("single_case", SINGLE_METHOD), ("mixture_case", MIXTURE_METHOD)):
            selection_rows.append(
                case_manifest_row(
                    row[method_key],
                    selection_group="paired",
                    selection=category,
                    pair_target_index=int(row["target_index"]),
                )
            )
    for method, method_cases in cases.items():
        for selection_name, position in representative_selections[method].items():
            selection_rows.append(
                case_manifest_row(
                    method_cases[position],
                    selection_group="representative",
                    selection=selection_name,
                )
            )
        best_cases, worst_cases = extreme_selections[method]
        for rank, case in enumerate(best_cases, start=1):
            selection_rows.append(
                case_manifest_row(
                    case,
                    selection_group="best-worst",
                    selection="best",
                    selection_rank=rank,
                )
            )
        for rank, case in enumerate(worst_cases, start=1):
            selection_rows.append(
                case_manifest_row(
                    case,
                    selection_group="best-worst",
                    selection="worst",
                    selection_rank=rank,
                )
            )
        method_by_target = case_map(method_cases)
        for rank, target_index in enumerate(random_target_indices, start=1):
            selection_rows.append(
                case_manifest_row(
                    method_by_target[target_index],
                    selection_group="random40",
                    selection="fixed_random",
                    selection_rank=rank,
                )
            )

    requested = str(args.figure_set)
    if requested in {"all", "paired"}:
        generated.extend(
            plot_paired_grid(
                paired_selected,
                PAIRED_SELECTION_ORDER[:4],
                output_dir / "paired_main4",
                args.formats,
            )
        )
        generated.extend(
            plot_paired_grid(
                paired_selected,
                PAIRED_SELECTION_ORDER,
                output_dir / "paired_supplement8",
                args.formats,
            )
        )
    if requested in {"all", "representative"}:
        for method, method_cases in cases.items():
            generated.extend(
                plot_four_case_figure(
                    method,
                    method_cases,
                    representative_selections[method],
                    output_dir / f"{method}_best_median_worst_random",
                    args.formats,
                )
            )
    if requested in {"all", "best-worst"}:
        for method in METHOD_LABELS:
            best_cases, worst_cases = extreme_selections[method]
            generated.extend(
                plot_best_worst_gallery(
                    method,
                    best_cases,
                    worst_cases,
                    output_dir
                    / f"{method}_best{len(best_cases)}_worst{len(worst_cases)}_gallery",
                    args.formats,
                )
            )
    if requested in {"all", "random40"}:
        for method, method_cases in cases.items():
            method_by_target = case_map(method_cases)
            gallery_cases = [method_by_target[target_index] for target_index in random_target_indices]
            generated.extend(
                plot_gallery(
                    method,
                    gallery_cases,
                    output_dir / f"{method}_random{gallery_count}_gallery",
                    args.formats,
                )
            )
    if requested in {"all", "individual"}:
        for target_index in sorted(individual_labels):
            row = paired_by_target[target_index]
            label = ", ".join(sorted(individual_labels[target_index]))
            generated.extend(
                plot_individual_page(
                    target_index,
                    row["single_case"],
                    row["mixture_case"],
                    label,
                    output_dir / "individual" / f"target_{target_index}",
                    args.formats,
                )
            )

    random_csv_path = output_dir / "random40_target_list.csv"
    write_csv(
        random_csv_path,
        [
            {"selection_rank": rank, "target_index": target_index}
            for rank, target_index in enumerate(random_target_indices, start=1)
        ],
    )
    selection_csv_path = output_dir / "selection_index.csv"
    write_csv(selection_csv_path, selection_rows)
    captions_path = output_dir / "figure_captions.md"
    write_text(
        captions_path,
        build_captions_markdown(
            paired_selected,
            gallery_count,
            int(args.extreme_count),
            int(args.seed),
        ),
    )
    paired_manifest: list[dict[str, Any]] = []
    for category in PAIRED_SELECTION_ORDER:
        row = paired_selected[category]
        paired_manifest.append(
            {
                "category": category,
                "target_index": int(row["target_index"]),
                "joint_score_pct": float(row["joint_score"]),
                "method_disagreement_pp": float(row["disagreement"]),
                "single_advantage_pp": float(row["single_advantage"]),
                "mixture_advantage_pp": float(row["mixture_advantage"]),
                "largest_absolute_calibration_gap_pp": float(row["calibration_gap"]),
                "methods": {
                    SINGLE_METHOD: case_manifest_row(row["single_case"]),
                    MIXTURE_METHOD: case_manifest_row(row["mixture_case"]),
                },
            }
        )
    selection_manifest_path = output_dir / "selection_manifest.json"
    selection_manifest = {
        "version": 2,
        "status": "frozen_for_visualization",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evidence_status": "single_seed_adjustment_test625_replay",
        "figure_set": requested,
        "run_dir": run_dir.as_posix(),
        "data": {
            "dataset": dataset_path.as_posix(),
            "dataset_sha256": sha256_file(dataset_path),
            "candidate_manifest": required[0].as_posix(),
            "candidate_manifest_sha256": sha256_file(required[0]),
            "replay_candidates": required[1].as_posix(),
            "replay_candidates_sha256": sha256_file(required[1]),
            "target_count_per_method": int(args.expected_target_count),
        },
        "selection_seed": int(args.seed),
        "selection_rules": {
            "candidate": "best real-MPB bands2-10 MRE among the replayed top-5 candidates",
            "paired": (
                "joint best/median/worst use the mean single/mixture real-MPB MRE; "
                "remaining categories rank their named diagnostic; duplicates use the next unused target"
            ),
            "representative": "method-specific best/median/worst by real-MPB MRE plus distinct fixed-seed random",
            "best_worst": "strict ascending best and strict descending worst real-MPB MRE",
            "random40": "one fixed-seed target list shared by both methods",
            "scope": "post hoc visualization only; no training, checkpoint, sampling, or ranking decisions",
        },
        "paired": paired_manifest,
        "representative": {
            method: {
                name: case_manifest_row(cases[method][position])
                for name, position in representative_selections[method].items()
            }
            for method in METHOD_LABELS
        },
        "best_worst": {
            method: {
                "best": [case_manifest_row(case) for case in extreme_selections[method][0]],
                "worst": [case_manifest_row(case) for case in extreme_selections[method][1]],
            }
            for method in METHOD_LABELS
        },
        "random40_target_indices": random_target_indices,
    }
    write_json(selection_manifest_path, selection_manifest)

    artifact_inputs = generated + [
        random_csv_path,
        selection_csv_path,
        captions_path,
        selection_manifest_path,
    ]
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    artifact_manifest = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": output_dir.as_posix(),
        "figure_set": requested,
        "artifacts": artifact_rows(output_dir, artifact_inputs),
    }
    write_json(artifact_manifest_path, artifact_manifest)
    summary = {
        "status": "completed",
        "output_dir": output_dir.as_posix(),
        "figure_set": requested,
        "paired_targets": {
            category: int(paired_selected[category]["target_index"])
            for category in PAIRED_SELECTION_ORDER
        },
        "random_target_count": gallery_count,
        "individual_target_count": len(individual_labels),
        "generated_figure_files": len(generated),
        "selection_manifest": selection_manifest_path.as_posix(),
        "artifact_manifest": artifact_manifest_path.as_posix(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
