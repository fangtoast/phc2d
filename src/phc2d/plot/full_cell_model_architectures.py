"""Render baseline, upgrade, and evidence diagrams for Full-Wh models.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Sequence


FORMATS = ("png", "svg")
PALETTE = {
    "ink": "#23343E",
    "muted": "#52636D",
    "line": "#34454F",
    "panel": "#F8FAFB",
    "panel_edge": "#CAD3D8",
    "gray": "#EDF1F3",
    "blue": "#D8EBF8",
    "blue_edge": "#2F6F9F",
    "blue_deep": "#5E9FCA",
    "green": "#E4F2E4",
    "green_edge": "#4F8A5B",
    "amber": "#F8EED8",
    "amber_edge": "#A9792A",
    "red": "#F5DEDE",
    "red_edge": "#A95050",
}


# Frozen 50k held-out test evidence copied from
# runs/20260810/full_cell_upgrade_50k_evaluation_20260810085944/native_test_summary.csv
# (the report asset manifest records the machine-source hash).
NATIVE_50K_RESULTS = {
    "mlp": {"B": (1.8006663875, 0.1112050996), "G": (1.7701052589, 0.8598117053)},
    "res_mlp_wide": {"B": (0.9574617357, 0.0434420332), "G": (0.3130832302, 0.0942777899)},
    "lifted_cnn": {"B": (1.4427070449, 0.0064355216), "G": (0.7011801394, 0.0212044698)},
    "fno2d_band_v2": {"B": (0.6296472412, 0.0015633953), "G": (0.1601556459, 0.0005452375)},
    "mlp_mixer_band_v1": {"B": (0.5927371387, 0.0314065067), "G": (0.0645968481, 0.0045726190)},
    "res_mlp_gated_v2": {"B": (0.8791637705, 0.0164797150), "G": (0.4001493988, 0.0387542318)},
    "convnext_band_v1": {"B": (0.3212159680, 0.0079016908), "G": (0.0500454428, 0.0041098419)},
    "tfno2d_band_v3": {"B": (0.3950067591, 0.0022996361), "G": (0.1019227225, 0.0015742027)},
    "dual_lane_band_operator_v1": {"B": (0.4664781551, 0.0076130084), "G": (0.1196069011, 0.0030374702)},
}


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "phc2d-full-cell-20260810",
            "pdf.fonttype": 42,
        }
    )


def _canvas(*, title: str, subtitle: str, figsize: tuple[float, float]) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=figsize)
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.set_axis_off()
    figure.patch.set_facecolor("white")
    axis.text(2, 97, title, fontsize=20, fontweight="bold", color=PALETTE["ink"], va="top")
    axis.text(2, 92.8, subtitle, fontsize=11, color=PALETTE["muted"], va="top")
    return figure, axis


def _panel(axis: Any, x: float, y: float, width: float, height: float, *, title: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.45,rounding_size=1.2",
            facecolor=PALETTE["panel"],
            edgecolor=PALETTE["panel_edge"],
            linewidth=1.2,
        )
    )
    axis.text(x + 1.8, y + height - 2.1, title, fontsize=12, fontweight="bold", color=PALETTE["ink"], va="top")


def _box(
    axis: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    detail: str = "",
    kind: str = "gray",
    fontsize: float = 10,
) -> None:
    from matplotlib.patches import FancyBboxPatch

    edge_key = {
        "blue": "blue_edge",
        "green": "green_edge",
        "amber": "amber_edge",
        "red": "red_edge",
    }.get(kind, "muted")
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.30,rounding_size=0.8",
            facecolor=PALETTE[kind],
            edgecolor=PALETTE[edge_key],
            linewidth=1.3,
        )
    )
    lines = [label]
    if detail:
        lines.extend(detail.split("\n"))
    center_y = y + height / 2 + (0.8 if detail else 0)
    axis.text(x + width / 2, center_y, lines[0], ha="center", va="center", fontsize=fontsize, fontweight="bold", color=PALETTE["ink"])
    if len(lines) > 1:
        axis.text(
            x + width / 2,
            center_y - 2.8,
            "\n".join(lines[1:]),
            ha="center",
            va="top",
            fontsize=max(7.5, fontsize - 2),
            linespacing=1.25,
            color=PALETTE["muted"],
        )


def _arrow(axis: Any, start: tuple[float, float], end: tuple[float, float], *, blue: bool = False, dashed: bool = False) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": PALETTE["blue_edge"] if blue else PALETTE["line"],
            "linewidth": 1.8 if blue else 1.5,
            "linestyle": "--" if dashed else "-",
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def _save_figure(figure: Any, output_base: Path, formats: Sequence[str]) -> list[Path]:
    import matplotlib.pyplot as plt

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in formats:
        path = output_base.with_suffix(f".{suffix}")
        metadata = {"Date": None, "Creator": "phc2d"} if suffix == "svg" else {"Software": "phc2d"}
        figure.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
            metadata=metadata,
        )
        if suffix == "svg":
            normalized = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
            path.write_text(normalized + "\n", encoding="utf-8", newline="\n")
        paths.append(path)
    plt.close(figure)
    return paths


def _plot_data_generation_pipeline(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        title="Dual-lane Full-cell data generation: analytic truth, synchronized views",
        subtitle="The MPB label is solved from the analytic periodic material function; 32/64/128 rasters are aligned learning and audit views, never the solver geometry.",
        figsize=(18.0, 10.4),
    )

    _panel(axis, 2, 67, 96, 23, title="A | Frozen deterministic structure plan")
    _box(axis, 5, 73, 13, 10, "config + seed", detail="master seed 20260806\nsample_id + family", kind="gray", fontsize=9.2)
    _arrow(axis, (18, 78), (23, 78))
    _box(axis, 23, 72, 18, 12, "stratified schedule", detail="B: 1/4 polar-1, 1/4 polar-2, 1/2 spectral\nG: 1/2 one-core, 1/2 two-core", kind="blue", fontsize=9.0)
    _arrow(axis, (41, 78), (46, 78))
    _box(axis, 46, 72, 18, 12, "analytic structure", detail="FullCellStructureSpec\nu(x,y), params, family", kind="green", fontsize=9.2)
    _arrow(axis, (64, 78), (69, 78))
    _box(axis, 69, 72, 12, 12, "audit gate", detail="256 x 256 stats\nreject / retry", kind="amber", fontsize=9.2)
    _arrow(axis, (81, 78), (86, 78))
    _box(axis, 86, 72, 9, 12, "freeze", detail="geometry hash\nsplit label", kind="green", fontsize=9.2)
    axis.text(50, 69.2, "Resume validates sample index, family, stratum, split and geometry identity before any solver work.", ha="center", fontsize=9.0, color=PALETTE["muted"])

    _panel(axis, 2, 34, 96, 30, title="B | Two synchronized paths from the same analytic structure")
    _box(axis, 5, 43.5, 13, 9, "u(x,y)", detail="periodic weight in [0,1]", kind="green", fontsize=9.5)
    _arrow(axis, (18, 48), (24, 53), blue=True)
    _arrow(axis, (18, 48), (24, 41))

    _box(axis, 24, 48, 18, 9, "physical path", detail="epsilon = 1 + 7.9 u\nlazy material callback", kind="blue", fontsize=9.3)
    _arrow(axis, (42, 52.5), (47, 52.5), blue=True)
    _box(axis, 47, 47, 17, 11, "real MPB", detail="square a=1, TE\nresolution 64, mesh 3", kind="blue", fontsize=9.3)
    _arrow(axis, (64, 52.5), (69, 52.5), blue=True)
    _box(axis, 69, 47, 12, 11, "full BZ", detail="9 x 9 = 81 k\nreduced coords", kind="gray", fontsize=9.2)
    _arrow(axis, (81, 52.5), (86, 52.5), blue=True)
    _box(axis, 86, 47, 9, 11, "Wh", detail="solve 14\nkeep 10", kind="green", fontsize=9.2)

    _box(axis, 24, 36.5, 18, 9, "representation path", detail="cell-center sampling\nfrom the same u(x,y)", kind="gray", fontsize=9.3)
    _arrow(axis, (42, 41), (47, 41))
    _box(axis, 47, 36.5, 17, 9, "three rasters", detail="weight + epsilon\n32 / 64 / 128", kind="amber", fontsize=9.3)
    _arrow(axis, (64, 41), (69, 41))
    _box(axis, 69, 36.5, 26, 9, "learning and audit only", detail="forward input: weight_64; resolution checks: 32/128", kind="gray", fontsize=9.2)
    axis.text(50, 32.4, "MPB does not read a PNG or a raster approximation of the structure.", ha="center", fontsize=9.2, color=PALETTE["amber_edge"], fontweight="bold")

    _panel(axis, 2, 7, 96, 24, title="C | Recoverable solve, fail-closed merge and final HDF5 contract")
    _box(axis, 5, 14.5, 14, 9, "one sample job", detail="all 81 k points\nidentity hashes", kind="gray", fontsize=9.2)
    _arrow(axis, (19, 19), (24, 19))
    _box(axis, 24, 14.5, 15, 9, "durable shards", detail="Wh hash + timing\nfailure ledger", kind="blue", fontsize=9.2)
    _arrow(axis, (39, 19), (44, 19))
    _box(axis, 44, 13, 16, 12, "atomic finalizer", detail="complete coverage\nzero failures\nfinite + band order", kind="amber", fontsize=9.2)
    _arrow(axis, (60, 19), (65, 19))
    _box(axis, 65, 12, 30, 14, "phc2d.full_cell.v1 HDF5", detail="structure_json / params / family | grids 32,64,128\nk_points[81,2] | Wh[N,81,10] | fixed train/val/test", kind="green", fontsize=9.2)
    axis.text(50, 9.4, "50k per lane: 40k train / 5k validation / 5k test; lane files and normalization remain separate.", ha="center", fontsize=9.2, color=PALETTE["muted"])
    return _save_figure(figure, output_base, formats)


def _plot_baseline_mlp_residual(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        title="Matched MLP baselines: the residual connection is the controlled difference",
        subtitle="For the 64 x 64 Full-Wh task, both models contain 2,046,506 parameters and six matched two-linear blocks.",
        figsize=(17.0, 8.6),
    )
    _panel(axis, 2, 53, 96, 37, title="A | Shared input, width, depth, activation and output")
    boxes = (
        (5, 12, "grid", "[B,64,64]"),
        (21, 12, "flatten", "4096 values"),
        (37, 15, "input projection", "4096 -> 256 -> GELU"),
        (57, 16, "matched blocks x 6", "256 -> 256 -> 256"),
        (78, 16, "output projection", "256 -> 810 -> [81,10]"),
    )
    for index, (x, width, label, detail) in enumerate(boxes):
        _box(axis, x, 67, width, 10, label, detail=detail, kind="blue" if index == 3 else "gray", fontsize=9.5)
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][0]
            _arrow(axis, (x + width, 72), (next_x, 72))
    axis.text(50, 59.5, "Optimizer, dropout (0.05), hidden width (256), depth (6), and parameter count are held fixed.", ha="center", fontsize=9.5, color=PALETTE["muted"])

    _panel(axis, 2, 8, 46, 37, title="B | mlp: plain transformation")
    _box(axis, 6, 23, 9, 9, "h_l", kind="gray")
    _arrow(axis, (15, 27.5), (21, 27.5))
    _box(axis, 21, 19.5, 18, 16, "F_l(h_l)", detail="Linear -> GELU -> Dropout\n-> Linear -> GELU", kind="blue", fontsize=9.5)
    axis.text(25, 13.3, "h_(l+1) = GELU(F_l(h_l))", fontsize=10, color=PALETTE["ink"], ha="center")

    _panel(axis, 52, 8, 46, 37, title="C | res_mlp_wide: identity-assisted transformation")
    _box(axis, 56, 23, 9, 9, "h_l", kind="gray")
    _arrow(axis, (65, 27.5), (71, 27.5))
    _box(axis, 71, 19.5, 18, 16, "F_l(h_l)", detail="Linear -> GELU -> Dropout\n-> Linear", kind="blue", fontsize=9.5)
    _arrow(axis, (60.5, 23), (60.5, 15), blue=True)
    _arrow(axis, (60.5, 15), (92, 23), blue=True)
    _box(axis, 89.5, 23, 6, 9, "+", detail="GELU", kind="green", fontsize=10)
    axis.text(75, 12.8, "h_(l+1) = GELU(h_l + F_l(h_l))", fontsize=10, color=PALETTE["ink"], ha="center")
    return _save_figure(figure, output_base, formats)


def _plot_baseline_lifted_cnn(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        title="lifted_cnn: a deliberately small local-convolution baseline",
        subtitle="Two 3 x 3 convolutions test whether local image features alone can support a fixed Full-Wh regression head.",
        figsize=(16.0, 7.8),
    )
    _panel(axis, 2, 45, 96, 45, title="A | End-to-end data flow")
    boxes = (
        (5, 12, "grid", "[B,1,64,64]"),
        (22, 15, "Conv 3 x 3", "1 -> 12; pad=1\nGELU"),
        (43, 15, "Conv 3 x 3", "12 -> 12; pad=1\nGELU"),
        (64, 13, "global average", "12 x H x W -> 12"),
        (83, 12, "dense head", "12 -> 810\n[81,10]"),
    )
    for index, (x, width, label, detail) in enumerate(boxes):
        _box(axis, x, 62, width, 13, label, detail=detail, kind="blue" if index in {1, 2} else "gray", fontsize=9.5)
        if index < len(boxes) - 1:
            _arrow(axis, (x + width, 68.5), (boxes[index + 1][0], 68.5))
    axis.text(50, 51.5, "11,958 parameters: low capacity is intentional, so this is a local-feature floor rather than a modern CNN.", ha="center", fontsize=9.5, color=PALETTE["muted"])

    _panel(axis, 2, 8, 96, 29, title="B | What this baseline can and cannot encode")
    _box(axis, 6, 17, 24, 10, "Local receptive fields", detail="translation-shared 3 x 3 filters", kind="green")
    _box(axis, 38, 17, 24, 10, "Global pooling", detail="size-agnostic summary; location is discarded", kind="amber")
    _box(axis, 70, 17, 24, 10, "Boundary handling", detail="zero padding, not periodic padding", kind="red")
    axis.text(50, 11.4, "Its failure cannot isolate one missing component: capacity, depth, receptive field, normalization and periodicity all change in ConvNeXt.", ha="center", fontsize=9.2, color=PALETTE["muted"])
    return _save_figure(figure, output_base, formats)


def _plot_baseline_fno(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        title="fno2d_band_v2: band-aware spectral baseline",
        subtitle="A four-block project-native FNO-style encoder mixes truncated Fourier modes, pointwise features and a low-rank global summary.",
        figsize=(18.0, 9.2),
    )
    _panel(axis, 2, 58, 96, 32, title="A | End-to-end encoder")
    boxes = (
        (5, 11, "grid", "[B,1,H,W]"),
        (21, 13, "1 x 1 lift", "1 -> 32"),
        (40, 21, "BandBlock x 4", "modes 8 x 8; band modes 5 x 5\nlow-rank width 8"),
        (68, 13, "global average", "32 x H x W -> 32"),
        (87, 9, "head", "32 -> 810"),
    )
    for index, (x, width, label, detail) in enumerate(boxes):
        _box(axis, x, 70, width, 11, label, detail=detail, kind="blue" if index == 2 else "gray", fontsize=9.2)
        if index < len(boxes) - 1:
            _arrow(axis, (x + width, 75.5), (boxes[index + 1][0], 75.5))

    _panel(axis, 2, 10, 96, 40, title="B | One BandBlock")
    _box(axis, 5, 26, 10, 9, "input", detail="h_l", kind="gray")
    _arrow(axis, (15, 30.5), (21, 39), blue=True)
    _arrow(axis, (15, 30.5), (21, 31))
    _arrow(axis, (15, 30.5), (21, 22), blue=True)
    _box(axis, 21, 36.5, 20, 8, "dual spectral weights", detail="rFFT2 full + band modes", kind="blue", fontsize=9)
    _box(axis, 21, 28.3, 20, 8, "pointwise branch", detail="1 x 1 Conv", kind="gray", fontsize=9)
    _box(axis, 21, 20.1, 20, 8, "low-rank global", detail="GAP -> 32:8:32", kind="gray", fontsize=9)
    _arrow(axis, (41, 39.5), (49, 32), blue=True)
    _arrow(axis, (41, 31.5), (49, 32))
    _box(axis, 49, 27.5, 14, 9, "band gate", detail="1 x 1 Conv -> sigmoid", kind="green", fontsize=9)
    _arrow(axis, (41, 23.5), (69, 28), blue=True)
    _arrow(axis, (63, 32), (69, 32))
    _box(axis, 69, 26, 12, 10, "merge", detail="+ global; 1 x 1", kind="blue", fontsize=9)
    _arrow(axis, (10, 26), (10, 15), blue=True)
    _arrow(axis, (10, 15), (87, 26), blue=True)
    _arrow(axis, (81, 31), (87, 31))
    _box(axis, 87, 26, 8, 10, "+ skip", detail="GELU", kind="green", fontsize=9)
    axis.text(50, 13, "406,218 parameters at 64 x 64 / 81 x 10. Variable tensor size is accepted, but zero-shot accuracy is not guaranteed.", ha="center", fontsize=9.2, color=PALETTE["muted"])
    return _save_figure(figure, output_base, formats)


def _plot_shared_heads(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        title="Shared Full-Wh interface and output heads",
        subtitle="Every upgraded encoder returns a structure latent z; the selected head always emits [B, Nk, Nb].",
        figsize=(16.0, 7.2),
    )
    _box(axis, 4, 72, 16, 10, "Encoder latent", detail="z [B, D]", kind="blue", fontsize=11)
    _arrow(axis, (20, 77), (27, 77))
    _box(axis, 27, 72, 18, 10, "Head selection", detail="dense | k-conditioned", kind="amber", fontsize=11)
    _arrow(axis, (45, 77), (53, 77))
    _box(axis, 53, 72, 17, 10, "Unified output", detail="Wh [B, Nk, Nb]", kind="green", fontsize=11)
    _box(axis, 75, 72, 20, 10, "Runtime contract", detail="k_points optional; lane_id explicit", fontsize=10)

    _panel(axis, 3, 8, 44, 54, title="A | dense head: fixed target topology")
    _box(axis, 7, 39, 10, 9, "z", detail="[B,D]", kind="blue")
    _arrow(axis, (17, 43.5), (22, 43.5))
    _box(axis, 22, 36, 17, 15, "LayerNorm(D)", detail="Linear(D, Nk x Nb)\nreshape", kind="gray")
    _arrow(axis, (39, 43.5), (43, 43.5))
    axis.text(25, 28, "Control role", fontsize=10.5, fontweight="bold", color=PALETTE["ink"])
    axis.text(25, 23.8, "Keeps the existing fixed Full-Wh output topology.", fontsize=9.5, color=PALETTE["muted"], ha="center")
    axis.text(25, 19.8, "k_points validates shape/provenance only; it does not change values.", fontsize=9.5, color=PALETTE["muted"], ha="center")
    axis.text(25, 14.6, "No claim of runtime k-grid generalization", fontsize=9.5, color=PALETTE["amber_edge"], ha="center", fontweight="bold")

    _panel(axis, 52, 8, 45, 54, title="B | k-conditioned head: vectorized (k, band) queries")
    _box(axis, 55, 46, 10, 8, "z", detail="[B,D]", kind="blue")
    _box(axis, 55, 35, 10, 8, "k", detail="[Nk,2]", kind="gray")
    _box(axis, 55, 24, 10, 8, "band id", detail="1...Nb", kind="gray")
    _arrow(axis, (65, 50), (71, 48))
    _arrow(axis, (65, 39), (71, 43), blue=True)
    _arrow(axis, (65, 28), (71, 38), blue=True)
    _box(axis, 71, 35, 13, 19, "Fusion", detail="Wz z + q(k,b)\nLayerNorm\n256-d hidden", kind="blue")
    _arrow(axis, (84, 44.5), (89, 44.5))
    _box(axis, 89, 39, 6, 11, "MLP", detail="256 -> 1", kind="green", fontsize=9)
    axis.text(75, 20.1, "phi(k): raw xy + Fourier features (1, 2, 4, 8)", fontsize=8.5, color=PALETTE["muted"], ha="center")
    axis.text(75, 16.7, "q(k,b): phi(k) + learned 32-d band embedding; vectorized over Nk x Nb", fontsize=8.5, color=PALETTE["muted"], ha="center")
    axis.text(75, 12.2, "Interface capability is not unseen-k evidence", fontsize=9.0, color=PALETTE["amber_edge"], ha="center", fontweight="bold")
    return _save_figure(figure, output_base, formats)


def _plot_mlp_upgrades(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        title="Full-Wh MLP upgrades: token mixing versus gated residual mixing",
        subtitle="Both use Pre-LN, layer scale and stochastic depth, but only the Mixer preserves an explicit spatial-token organization.",
        figsize=(17.0, 9.0),
    )
    _panel(axis, 2, 51, 96, 39, title="A | mlp_mixer_band_v1")
    x_positions = (5, 20, 35, 50, 68, 84)
    boxes = (
        (12, "grid", "[B,C,H,W]"),
        (12, "area pool", "configured Ph x Pw"),
        (12, "tokens", "[B,T,C]"),
        (14, "Linear + pos", "C -> 256"),
        (12, "Mixer x 8", "Pre-LN residual"),
        (11, "LN + mean", "z [B,256]"),
    )
    for index, (x, (width, label, detail)) in enumerate(zip(x_positions, boxes)):
        _box(axis, x, 69, width, 9, label, detail=detail, kind="blue" if index in {3, 4} else "gray", fontsize=9.5)
        if index < len(x_positions) - 1:
            _arrow(axis, (x + width, 73.5), (x_positions[index + 1], 73.5))
    _box(axis, 21, 55.5, 27, 9, "Token residual", detail="LN -> transpose -> T:128:T MLP -> layer scale -> DropPath", kind="green", fontsize=9)
    _box(axis, 54, 55.5, 27, 9, "Channel residual", detail="LN -> 256:512:256 MLP -> layer scale -> DropPath", kind="green", fontsize=9)
    _arrow(axis, (74, 69), (67.5, 64.5), blue=True)
    axis.text(88.5, 57.2, "Variable input grid\nvia adaptive tokenization", fontsize=9.0, color=PALETTE["green_edge"], ha="center", fontweight="bold")

    _panel(axis, 2, 7, 96, 38, title="B | res_mlp_gated_v2")
    x_positions = (5, 22, 41, 63, 84)
    boxes = (
        (13, "flattened grid", "[B,C*H*W]"),
        (15, "input projection", "Linear -> 320 -> GELU"),
        (17, "gated block x 8", "Pre-LN residual"),
        (15, "LayerNorm", "z [B,320]"),
        (11, "band head", "[B,Nk,Nb]"),
    )
    for index, (x, (width, label, detail)) in enumerate(zip(x_positions, boxes)):
        _box(axis, x, 25, width, 9, label, detail=detail, kind="blue" if index == 2 else "gray", fontsize=9.5)
        if index < len(x_positions) - 1:
            _arrow(axis, (x + width, 29.5), (x_positions[index + 1], 29.5))
    _box(axis, 37, 10.5, 38, 9, "GatedResidualMLPBlock", detail="u=LN(h); GELU(Wg u) x Wv u -> dropout -> Wo -> alpha -> DropPath -> +h", kind="green", fontsize=9)
    _arrow(axis, (49.5, 25), (49.5, 19.5), blue=True)
    axis.text(86, 14.3, "Fixed flattened input shape\nmust match checkpoint", fontsize=9.5, color=PALETTE["amber_edge"], ha="center", fontweight="bold")
    return _save_figure(figure, output_base, formats)


def _plot_convnext(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        title="convnext_band_v1: periodic multi-stage convolutional encoder",
        subtitle="The upgrade replaces the two-layer Lifted CNN with deep residual stages while keeping cell boundaries circular.",
        figsize=(17.0, 8.6),
    )
    _panel(axis, 2, 55, 96, 35, title="A | End-to-end encoder")
    x_positions = (5, 20, 37, 53, 70, 86)
    boxes = (
        (11, "grid", "[B,C,H,W]"),
        (13, "patch stem", "Conv k=4, s=4; C->48"),
        (12, "stage 1", "48 ch x 2"),
        (13, "stage 2", "96 ch x 3"),
        (12, "stage 3", "192 ch x 6"),
        (9, "GAP + LN", "z [B,192]"),
    )
    for index, (x, (width, label, detail)) in enumerate(zip(x_positions, boxes)):
        _box(axis, x, 69, width, 9, label, detail=detail, kind="blue" if index in {2, 3, 4} else "gray", fontsize=9)
        if index < len(x_positions) - 1:
            _arrow(axis, (x + width, 73.5), (x_positions[index + 1], 73.5))
    axis.text(45.5, 62.5, "GroupNorm -> average pool 2 -> 1x1 Conv: 48 -> 96", fontsize=8.6, color=PALETTE["muted"], ha="center")
    axis.text(70, 62.5, "GroupNorm -> average pool 2 -> 1x1 Conv: 96 -> 192", fontsize=8.6, color=PALETTE["muted"], ha="center")

    _panel(axis, 2, 8, 96, 39, title="B | One ConvNeXt block")
    _box(axis, 5, 24, 9, 9, "input", detail="X_l", kind="gray")
    _arrow(axis, (14, 28.5), (20, 28.5))
    _box(axis, 20, 21, 16, 15, "spatial mixing", detail="circular pad(3)\ndepthwise Conv 7x7", kind="blue", fontsize=9.5)
    _arrow(axis, (36, 28.5), (42, 28.5))
    _box(axis, 42, 21, 12, 15, "channels-last", detail="LayerNorm", kind="gray", fontsize=9.5)
    _arrow(axis, (54, 28.5), (60, 28.5))
    _box(axis, 60, 21, 18, 15, "channel MLP", detail="D -> 4D -> D\nGELU", kind="blue", fontsize=9.5)
    _arrow(axis, (78, 28.5), (84, 28.5))
    _box(axis, 84, 21, 11, 15, "residual", detail="scale 1e-6\nDropPath", kind="green", fontsize=9.5)
    _arrow(axis, (14, 25), (18, 15), blue=True)
    _arrow(axis, (18, 15), (89.5, 21), blue=True)
    axis.text(50, 12.5, "Circular padding prevents artificial zero-valued interfaces at periodic unit-cell boundaries.", fontsize=9.5, color=PALETTE["muted"], ha="center")
    return _save_figure(figure, output_base, formats)


def _plot_tfno_dual_lane(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        title="TFNO-style Full-Wh operators and dual-lane conditioning",
        subtitle="tfno2d_band_v3 and dual_lane_band_operator_v1 share the same factorized encoder; the latter adds lane FiLM and lane-specific normalization.",
        figsize=(18.0, 10.2),
    )
    _panel(axis, 2, 65, 96, 25, title="A | Shared factorized operator encoder")
    x_positions = (5, 22, 40, 64, 84)
    boxes = (
        (12, "grid", "[B,C,H,W]"),
        (13, "1x1 lifting", "C -> 64"),
        (18, "OperatorBlock x 6", "modes 12x12; rank 0.75"),
        (15, "GroupNorm + GELU", "spatial mean"),
        (11, "band head", "[B,Nk,Nb]"),
    )
    for index, (x, (width, label, detail)) in enumerate(zip(x_positions, boxes)):
        _box(axis, x, 74, width, 9, label, detail=detail, kind="blue" if index == 2 else "gray", fontsize=9.5)
        if index < len(x_positions) - 1:
            _arrow(axis, (x + width, 78.5), (x_positions[index + 1], 78.5))

    _panel(axis, 2, 28.5, 61, 31.5, title="B | FactorizedOperatorBlock: three learned branches")
    _box(axis, 5, 39, 10, 9, "Pre-GN", detail="u [B,64,H,W]", kind="gray")
    _arrow(axis, (15, 43.5), (21, 49), blue=True)
    _arrow(axis, (15, 43.5), (21, 43))
    _arrow(axis, (15, 43.5), (21, 37), blue=True)
    _box(axis, 21, 46.8, 19, 7.2, "spectral", detail="rFFT2; +/- first axis", kind="blue", fontsize=9)
    _box(axis, 21, 39.4, 19, 7.2, "local", detail="circular Conv 3x3", kind="gray", fontsize=9)
    _box(axis, 21, 32.0, 19, 7.2, "channel", detail="1x1 -> GELU -> 1x1", kind="gray", fontsize=9)
    _arrow(axis, (40, 50), (46, 44), blue=True)
    _arrow(axis, (40, 43.5), (46, 44))
    _arrow(axis, (40, 37), (46, 44), blue=True)
    _box(axis, 46, 39, 13, 10, "soft gates", detail="sigmoid x 3\n+ residual", kind="green", fontsize=9)
    axis.text(32, 30.5, "Complex Tucker core + input/output/y/x factors", fontsize=8.6, color=PALETTE["muted"], ha="center")

    _panel(axis, 66, 28.5, 32, 31.5, title="C | Dual-lane FiLM")
    _box(axis, 69, 46, 10, 8, "lane id", detail="B=0 | G=1", kind="amber", fontsize=9)
    _arrow(axis, (79, 50), (83, 50), blue=True)
    _box(axis, 83, 46, 11, 8, "embedding", detail="2 x 16", kind="blue", fontsize=9)
    _arrow(axis, (88.5, 46), (88.5, 41), blue=True)
    _box(axis, 72, 33.5, 22, 8, "per-block FiLM", detail="u x (1 + gamma) + beta", kind="green", fontsize=9)
    axis.text(82, 30.8, "gamma/beta [B,64,1,1]", fontsize=8.5, color=PALETTE["muted"], ha="center")

    _panel(axis, 2, 7, 96, 19, title="D | Identity and evidence boundary")
    axis.text(5, 18.5, "tfno2d_band_v3", fontsize=10.5, fontweight="bold", color=PALETTE["blue_edge"])
    axis.text(20, 18.5, "project-native Tucker-factorized FNO-style encoder; not official NeuralOperator TFNO code", fontsize=9.5, color=PALETTE["muted"])
    axis.text(5, 14, "dual_lane operator", fontsize=10.5, fontweight="bold", color=PALETTE["green_edge"])
    axis.text(20, 14, "shared encoder/head; lane-specific train statistics and checkpoint metadata remain mandatory", fontsize=9.5, color=PALETTE["muted"])
    axis.text(5, 9.5, "Evidence gate", fontsize=10.5, fontweight="bold", color=PALETTE["amber_edge"])
    axis.text(20, 9.5, "native held-out surrogate evidence does not prove resolution invariance, cross-lane equivalence, or real-MPB physics", fontsize=9.5, color=PALETTE["muted"])
    return _save_figure(figure, output_base, formats)


def _plot_native_50k_comparison(output_base: Path, formats: Sequence[str]) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    _style()
    labels = [
        "MLP",
        "ResMLP",
        "Lifted CNN",
        "FNO v2",
        "Mixer",
        "Gated ResMLP",
        "ConvNeXt",
        "TFNO v3",
        "Dual-lane",
    ]
    keys = list(NATIVE_50K_RESULTS)
    colors = [PALETTE["muted"]] * 4 + [PALETTE["blue_deep"]] * 5
    figure, axes = plt.subplots(1, 2, figsize=(17.0, 8.0))
    figure.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.24, wspace=0.10)
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "Native 50k held-out test: baselines and upgrades on one frozen contract",
        fontsize=19,
        fontweight="bold",
        color=PALETTE["ink"],
        y=0.98,
    )
    figure.text(
        0.5,
        0.915,
        "Mean +/- standard deviation across three seeds; lower MRE is better. The vertical divider separates four baselines from five upgrades.",
        ha="center",
        fontsize=10.5,
        color=PALETTE["muted"],
    )
    x = np.arange(len(keys))
    for axis, lane, title in zip(axes, ("B", "G"), ("Binary lane (B)", "Graded lane (G)")):
        means = [NATIVE_50K_RESULTS[key][lane][0] for key in keys]
        stds = [NATIVE_50K_RESULTS[key][lane][1] for key in keys]
        bars = axis.bar(x, means, yerr=stds, color=colors, edgecolor="white", linewidth=0.8, capsize=3)
        axis.axvline(3.5, color=PALETTE["panel_edge"], linewidth=1.4, linestyle="--")
        axis.set_title(title, fontsize=14, fontweight="bold", color=PALETTE["ink"], pad=12)
        axis.set_ylabel("held-out test MRE (%)", color=PALETTE["ink"])
        axis.set_xticks(x, labels, rotation=38, ha="right")
        axis.grid(axis="y", color=PALETTE["panel_edge"], alpha=0.55, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(1.5, axis.get_ylim()[1] * 0.97, "baselines", ha="center", va="top", fontsize=9, color=PALETTE["muted"])
        axis.text(6.0, axis.get_ylim()[1] * 0.97, "upgrades", ha="center", va="top", fontsize=9, color=PALETTE["blue_edge"])
        for bar, value in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.7,
                color=PALETTE["ink"],
            )
    figure.text(
        0.5,
        0.035,
        "Evidence boundary: same held-out test and split; this comparison is surrogate accuracy, not real-MPB replay or an isolated component ablation.",
        ha="center",
        fontsize=9.5,
        color=PALETTE["amber_edge"],
        fontweight="bold",
    )
    return _save_figure(figure, output_base, formats)


def _plot_inverse_stage_a_models(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        title="Full-cell inverse Stage A: deterministic and mode-conditioned parameter networks",
        subtitle="Each analytic structure variant owns a separate output contract; the networks predict codec tensors, never a padded 580-value union.",
        figsize=(18.0, 10.0),
    )
    _panel(axis, 2, 67, 96, 23, title="A | Shared condition and variant-specific output")
    _box(axis, 5, 74, 14, 9, "target Wh", detail="[B,81,10]", kind="gray")
    _arrow(axis, (19, 78.5), (25, 78.5))
    _box(axis, 25, 72, 22, 13, "train-only normalization", detail="per (k,band) mean/std\nflatten -> 810", kind="blue", fontsize=9.5)
    _arrow(axis, (47, 78.5), (54, 78.5))
    _box(axis, 54, 72, 16, 13, "inverse network", detail="Direct | Single | Mixture", kind="amber", fontsize=9.5)
    _arrow(axis, (70, 78.5), (77, 78.5))
    _box(axis, 77, 72, 18, 13, "variant codec tensor", detail="width d_v is codec-specific\nphysical DoF: 11 | 22 | 289 | 57 | 63", kind="green", fontsize=9.0)

    _panel(axis, 2, 37, 46, 23, title="B | direct_inverse_mlp")
    _box(axis, 6, 44, 10, 8, "Wh", detail="810", kind="gray")
    _arrow(axis, (16, 48), (21, 48))
    _box(axis, 21, 41.5, 17, 13, "MLP", detail="810 -> 512 -> 512\n2 x GELU", kind="blue", fontsize=9.5)
    _arrow(axis, (38, 48), (42, 48))
    _box(axis, 42, 44, 4, 8, "p", detail="d_v", kind="green", fontsize=9)
    axis.text(25, 39.3, "Deterministic: one target produces one candidate.", ha="center", fontsize=8.8, color=PALETTE["muted"])

    _panel(axis, 52, 37, 46, 23, title="C | single_mode_generator")
    _box(axis, 55, 46, 8, 7, "Wh", detail="810", kind="gray")
    _box(axis, 55, 38.8, 8, 5.5, "z (32)", kind="amber", fontsize=8.5)
    _arrow(axis, (63, 49.5), (68, 50))
    _box(axis, 68, 45, 11, 10, "condition", detail="810:512:512", kind="blue", fontsize=9)
    axis.text(65.5, 41.5, "+", fontsize=14, fontweight="bold", color=PALETTE["line"], ha="center", va="center")
    _box(axis, 68, 38.8, 11, 5.5, "mode (1 x 512)", kind="gray", fontsize=8)
    _arrow(axis, (79, 50), (83, 50))
    _arrow(axis, (79, 41.5), (83, 45), blue=True)
    _box(axis, 83, 42, 12, 13, "decoder", detail="(512+512+32)\n4 x 512 -> d_v", kind="green", fontsize=9)

    _panel(axis, 2, 8, 96, 22, title="D | mixture_conditional_generator and hard-decoder boundary")
    _box(axis, 5, 15, 15, 8, "condition encoder", detail="810 -> 512 -> 512", kind="blue", fontsize=9)
    _box(axis, 25, 15, 14, 8, "latent", detail="z [B,32]", kind="amber", fontsize=9)
    _box(axis, 44, 15, 15, 8, "mode embedding", detail="4 x 512; round-robin", kind="blue", fontsize=9)
    axis.text(22.5, 19, "+", fontsize=15, fontweight="bold", color=PALETTE["line"], ha="center", va="center")
    axis.text(41.5, 19, "+", fontsize=15, fontweight="bold", color=PALETTE["line"], ha="center", va="center")
    _arrow(axis, (59, 19), (65, 19), blue=True)
    _box(axis, 65, 13, 13, 12, "decoder", detail="4 x 512 -> d_v", kind="green", fontsize=9)
    _arrow(axis, (78, 19), (83, 19))
    _box(axis, 83, 13, 12, 12, "codec decode", detail="hard analytic geometry", kind="red", fontsize=9)
    axis.text(50, 10.2, "Soft decode is used only for differentiable objectives; candidate freeze and MPB always use the hard analytic decoder.", ha="center", fontsize=9.1, color=PALETTE["amber_edge"], fontweight="bold")
    return _save_figure(figure, output_base, formats)


FIGURES: tuple[tuple[str, Callable[[Path, Sequence[str]], list[Path]]], ...] = (
    ("figure_2_1_data_generation_pipeline", _plot_data_generation_pipeline),
    ("figure_4_1_baseline_mlp_residual", _plot_baseline_mlp_residual),
    ("figure_4_2_baseline_lifted_cnn", _plot_baseline_lifted_cnn),
    ("figure_4_3_baseline_fno2d_band", _plot_baseline_fno),
    ("figure_5_1_full_wh_heads", _plot_shared_heads),
    ("figure_5_2_mlp_upgrades", _plot_mlp_upgrades),
    ("figure_5_3_convnext_band", _plot_convnext),
    ("figure_5_4_tfno_dual_lane", _plot_tfno_dual_lane),
    ("figure_6_1_native_50k_model_comparison", _plot_native_50k_comparison),
    ("figure_7_1_inverse_stage_a_models", _plot_inverse_stage_a_models),
)


def build_full_cell_model_architecture_figures(
    *, output_dir: Path, formats: Sequence[str] = FORMATS
) -> dict[str, Any]:
    """Render all Full-Wh baseline, upgrade, and comparison diagrams."""

    invalid = sorted(set(formats) - set(FORMATS))
    if invalid:
        raise ValueError(f"Unsupported Full-Wh figure formats: {invalid}")
    _style()
    output_dir = Path(output_dir)
    generated: list[Path] = []
    for stem, builder in FIGURES:
        generated.extend(builder(output_dir / stem, formats))
    return {
        "status": "full_cell_model_architecture_figures_completed",
        "figure_count": len(FIGURES),
        "artifact_count": len(generated),
        "artifacts": [path.as_posix() for path in generated],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=list(FORMATS), choices=FORMATS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_full_cell_model_architecture_figures(
        output_dir=args.output_dir,
        formats=tuple(args.formats),
    )
    for path in result["artifacts"]:
        print(path)


if __name__ == "__main__":
    main()
