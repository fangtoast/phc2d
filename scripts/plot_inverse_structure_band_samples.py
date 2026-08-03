"""
Plot inverse-design structure and band overlays from saved replay artifacts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from phc2d.data import lift_rho_to_grid


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_bg": "#DDF3DE",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "orange": "#D95F02",
}

STRUCTURE_CMAP = ListedColormap(["white", PALETTE["blue_secondary"]])


@dataclass(frozen=True)
class ReplayCase:
    name: str
    local_dir: Path
    mpb_dir: Path
    dataset_path: Path
    target_indices_path: Path
    generated_structures_path: Path
    target_bands_path: Path
    replay_bands_path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def discover_replay_cases(source_run: Path, root: Path) -> list[ReplayCase]:
    """Find local inverse outputs with paired MPB replay bands."""
    cases: list[ReplayCase] = []
    for local_dir in sorted(source_run.glob("*_local")):
        results_path = local_dir / "results.json"
        if not results_path.exists():
            continue
        payload = _load_json(results_path)
        mpb_dir = local_dir.with_name(local_dir.name[: -len("_local")] + "_mpb")
        replay_bands_path = mpb_dir / "replay_bands.npy"
        required = {
            "data": payload.get("data"),
            "target_indices_file": payload.get("target_indices_file"),
            "generated_structures_file": payload.get("generated_structures_file"),
            "target_bands_file": payload.get("target_bands_file"),
        }
        if not replay_bands_path.exists() or any(v is None for v in required.values()):
            continue
        cases.append(
            ReplayCase(
                name=local_dir.name[: -len("_local")],
                local_dir=local_dir,
                mpb_dir=mpb_dir,
                dataset_path=_resolve_path(str(required["data"]), root),
                target_indices_path=_resolve_path(str(required["target_indices_file"]), root),
                generated_structures_path=_resolve_path(str(required["generated_structures_file"]), root),
                target_bands_path=_resolve_path(str(required["target_bands_file"]), root),
                replay_bands_path=replay_bands_path,
            )
        )
    if not cases:
        raise FileNotFoundError(f"No paired *_local/*_mpb replay cases found under {source_run}")
    return cases


def rho_to_image(rho: np.ndarray) -> np.ndarray:
    """Convert a saved rho vector/grid to a displayable binary grid."""
    arr = np.asarray(rho, dtype=np.float64)
    if arr.ndim == 2:
        return (arr >= 0.5).astype(float)
    try:
        return (lift_rho_to_grid(arr.reshape(-1)) >= 0.5).astype(float)
    except Exception:
        flat = arr.reshape(-1)
        side = int(np.sqrt(flat.size))
        if side * side != flat.size:
            raise ValueError(f"Cannot display rho with flattened size {flat.size}") from None
        return (flat.reshape(side, side) >= 0.5).astype(float)


def mean_relative_error_pct(y_true: np.ndarray, y_pred: np.ndarray, band_slice: slice | None = None) -> float:
    """Return mean relative error in percent with a small denominator floor."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if band_slice is not None:
        true = true[:, band_slice]
        pred = pred[:, band_slice]
    denom = np.maximum(np.abs(true), 1e-6)
    return float(100.0 * np.mean(np.abs(pred - true) / denom))


def draw_structure(ax: plt.Axes, rho: np.ndarray, title: str) -> None:
    image = rho_to_image(rho)
    ax.imshow(image, cmap=STRUCTURE_CMAP, origin="lower", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#333333")


def draw_bands(ax: plt.Axes, target: np.ndarray, replay: np.ndarray) -> None:
    target = np.asarray(target, dtype=np.float64)
    replay = np.asarray(replay, dtype=np.float64)
    band_count = min(target.shape[-1], replay.shape[-1])
    x = np.arange(min(target.shape[0], replay.shape[0]))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, band_count))
    for band in range(band_count):
        ax.plot(x, target[: x.size, band], color=colors[band], linewidth=1.0, alpha=0.95)
        ax.plot(x, replay[: x.size, band], color=colors[band], linewidth=1.0, linestyle="--", alpha=0.82)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_xlabel("k path", fontsize=10)
    ax.set_ylabel(r"$\omega a/(2\pi c)$", fontsize=10)
    ax.set_xticks([x[0], x[len(x) // 3], x[(2 * len(x)) // 3], x[-1]])
    ax.set_xticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$"])
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.tick_params(labelsize=9)


def plot_sample(
    *,
    out_path: Path,
    case_name: str,
    sample_pos: int,
    target_index: int,
    true_rho: np.ndarray,
    generated_rho: np.ndarray,
    target_bands: np.ndarray,
    replay_bands: np.ndarray,
) -> dict[str, str | int | float]:
    band_count = int(min(target_bands.shape[-1], replay_bands.shape[-1]))
    bands_2_end = slice(1, band_count) if band_count > 1 else None
    mre_all = mean_relative_error_pct(target_bands, replay_bands)
    mre_band1 = mean_relative_error_pct(target_bands, replay_bands, slice(0, 1))
    mre_bands_2_end = mean_relative_error_pct(target_bands, replay_bands, bands_2_end) if bands_2_end else float("nan")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.0, 3.8), facecolor=PALETTE["green_bg"])
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[0.82, 3.15],
        height_ratios=[1.0, 1.0],
        left=0.055,
        right=0.985,
        top=0.88,
        bottom=0.16,
        wspace=0.18,
        hspace=0.20,
    )
    ax_true = fig.add_subplot(grid[0, 0])
    ax_pred = fig.add_subplot(grid[1, 0])
    ax_band = fig.add_subplot(grid[:, 1])

    draw_structure(ax_true, true_rho, "true structure")
    draw_structure(ax_pred, generated_rho, "predicted structure")
    draw_bands(ax_band, target_bands, replay_bands)
    ax_band.plot([], [], color="black", linewidth=1.1, label="target bands")
    ax_band.plot([], [], color="black", linewidth=1.1, linestyle="--", label="pred/replay bands")
    ax_band.legend(loc="lower right", frameon=True, fontsize=8)

    fig.suptitle(
        (
            f"{case_name} | sample={sample_pos} | target_index={target_index} | "
            f"MRE b1={mre_band1:.2f}%, b2-{band_count}={mre_bands_2_end:.2f}%"
        ),
        fontsize=10.5,
        y=0.965,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return {
        "case": case_name,
        "sample_pos": int(sample_pos),
        "target_index": int(target_index),
        "figure": str(out_path),
        "mre_bands_1_to_end_pct": mre_all,
        "mre_band1_pct": mre_band1,
        "mre_bands_2_to_end_pct": mre_bands_2_end,
    }


def sample_plan(cases: Sequence[ReplayCase], n_samples: int, seed: int) -> list[tuple[ReplayCase, int]]:
    rng = np.random.default_rng(seed)
    sizes = []
    for case in cases:
        target_indices = np.load(case.target_indices_path, allow_pickle=False)
        sizes.append(int(target_indices.shape[0]))
    total = int(sum(sizes))
    if total <= 0:
        raise ValueError("No samples available in replay cases.")
    flat_choices = rng.choice(total, size=min(n_samples, total), replace=False)
    plan: list[tuple[ReplayCase, int]] = []
    offsets = np.cumsum([0, *sizes])
    for flat_idx in flat_choices:
        case_idx = int(np.searchsorted(offsets, flat_idx, side="right") - 1)
        sample_pos = int(flat_idx - offsets[case_idx])
        plan.append((cases[case_idx], sample_pos))
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("runs/20260612/inverse_dispersion_formal_replay_20260612_225928"),
        help="Run directory containing paired *_local and *_mpb inverse replay cases.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated figures.")
    parser.add_argument("--n-samples", type=int, default=40, help="Number of random sample figures to draw.")
    parser.add_argument("--seed", type=int, default=20260702, help="Random seed controlling sample selection.")
    args = parser.parse_args(argv)

    root = _repo_root()
    source_run = _resolve_path(args.source_run, root)
    output_dir = _resolve_path(args.output_dir, root)
    figures_dir = output_dir / "figures"
    cases = discover_replay_cases(source_run, root)
    plan = sample_plan(cases, int(args.n_samples), int(args.seed))

    dataset_cache: dict[Path, np.lib.npyio.NpzFile] = {}
    case_cache: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, str | int | float]] = []
    for global_pos, (case, sample_pos) in enumerate(plan, start=1):
        if case.dataset_path not in dataset_cache:
            dataset_cache[case.dataset_path] = np.load(case.dataset_path, allow_pickle=True)
        if case.name not in case_cache:
            case_cache[case.name] = {
                "target_indices": np.load(case.target_indices_path, allow_pickle=False),
                "generated_structures": np.load(case.generated_structures_path, allow_pickle=False),
                "target_bands": np.load(case.target_bands_path, allow_pickle=False),
                "replay_bands": np.load(case.replay_bands_path, allow_pickle=False),
            }
        dataset = dataset_cache[case.dataset_path]
        arrays = case_cache[case.name]
        target_index = int(arrays["target_indices"][sample_pos])
        figure_name = f"{global_pos:02d}_{case.name}_sample_{sample_pos:03d}_target_{target_index:06d}.png"
        rows.append(
            plot_sample(
                out_path=figures_dir / figure_name,
                case_name=case.name,
                sample_pos=sample_pos,
                target_index=target_index,
                true_rho=np.asarray(dataset["rho"][target_index]),
                generated_rho=np.asarray(arrays["generated_structures"][sample_pos]),
                target_bands=np.asarray(arrays["target_bands"][sample_pos]),
                replay_bands=np.asarray(arrays["replay_bands"][sample_pos]),
            )
        )

    manifest = {
        "source_run": str(source_run),
        "output_dir": str(output_dir),
        "n_samples_requested": int(args.n_samples),
        "n_samples_written": len(rows),
        "seed": int(args.seed),
        "cases": [
            {
                "name": c.name,
                "local_dir": str(c.local_dir),
                "mpb_dir": str(c.mpb_dir),
                "dataset_path": str(c.dataset_path),
            }
            for c in cases
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with (output_dir / "sample_index.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
