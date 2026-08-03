"""Generate the detailed BISON architecture figure for the manuscript.

Figure contract
---------------
Core conclusion: BISON is a concrete full-band spectral operator whose
two-scale Fourier mixing, channel gate, low-rank global context, and residual
update can be traced directly to the implemented BandBlock.
Figure archetype: schematic-led composite.
Target/output: double-column conference figure; SVG, PDF, PNG, and TIFF.
Backend: Python (Matplotlib) only.
Final size: 183 mm by 119 mm.
Panel map: (a) end-to-end tensor flow; (b) one BandBlock expanded.
Evidence hierarchy: implementation-level tensor flow is primary; compact
equations are supporting evidence. No statistical panel is present.
Source data: model implementation and the formal A1 configuration.
Image integrity: vector primitives only; no quantitative raster manipulation.
Reviewer risk: the diagram must not imply that BISON is an MPB/Maxwell solver,
or that the query operator and inverse generator are inside this network.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


# Mandatory publication and editable-vector settings.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 6.4
plt.rcParams["mathtext.fontset"] = "dejavusans"


HERE = Path(__file__).resolve().parent
OUT_BASE = HERE.parent / "figures" / "bison_architecture_detailed"

INK = "#17324D"
EDGE = "#31566F"
BLUE = "#2A6FAD"
BLUE_MID = "#8FBBDD"
BLUE_LIGHT = "#E8F2FA"
PURPLE_LIGHT = "#EEEAF6"
ORANGE_LIGHT = "#FBEBDD"
GREEN_LIGHT = "#E8F3E8"
GRAY_LIGHT = "#F3F6F8"
PANEL_BG = "#F8FAFC"
PANEL_EDGE = "#C7D3DC"
WHITE = "#FFFFFF"


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str | None = None,
    *,
    face: str = WHITE,
    edge: str = EDGE,
    title_size: float = 6.8,
    subtitle_size: float = 5.3,
    lw: float = 0.85,
    radius: float = 0.8,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.25,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.60,
            title,
            ha="center",
            va="center",
            color=INK,
            fontsize=title_size,
            fontweight="bold",
            zorder=zorder + 1,
        )
        ax.text(
            x + w / 2,
            y + h * 0.31,
            subtitle,
            ha="center",
            va="center",
            color=EDGE,
            fontsize=subtitle_size,
            linespacing=1.15,
            zorder=zorder + 1,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            color=INK,
            fontsize=title_size,
            fontweight="bold",
            linespacing=1.15,
            zorder=zorder + 1,
        )
    return patch


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    lw: float = 0.9,
    style: str = "-|>",
    mutation: float = 8.0,
    zorder: int = 5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def add_route(
    ax,
    points: list[tuple[float, float]],
    *,
    color: str = INK,
    lw: float = 0.85,
    dashed: bool = False,
    arrow: bool = True,
    zorder: int = 4,
) -> None:
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    if arrow:
        patch = FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=7.5,
            linewidth=lw,
            linestyle="--" if dashed else "-",
            color=color,
            zorder=zorder,
        )
    else:
        patch = PathPatch(
            path,
            fill=False,
            linewidth=lw,
            linestyle="--" if dashed else "-",
            edgecolor=color,
            zorder=zorder,
        )
    ax.add_patch(patch)


def add_operator_node(ax, x: float, y: float, symbol: str, face: str = BLUE) -> None:
    ax.add_patch(Circle((x, y), 1.42, facecolor=face, edgecolor=EDGE, linewidth=0.8, zorder=6))
    ax.text(x, y - 0.03, symbol, ha="center", va="center", color=WHITE, fontsize=9.5, fontweight="bold", zorder=7)


def add_group(ax, x: float, y: float, w: float, h: float, label: str, face: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.20,rounding_size=0.9",
            facecolor=face,
            edgecolor=PANEL_EDGE,
            linewidth=0.7,
            zorder=1,
        )
    )
    ax.text(x + 1.0, y + h - 1.1, label, ha="left", va="top", color=EDGE, fontsize=5.5, fontweight="bold", zorder=2)


def draw_input_grid(ax, x: float, y: float, size: float) -> None:
    pattern = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ]
    cell = size / 5
    ax.add_patch(Rectangle((x + 0.8, y + 0.8), size, size, facecolor="#DDE5EA", edgecolor=EDGE, linewidth=0.55, zorder=1))
    ax.add_patch(Rectangle((x + 0.4, y + 0.4), size, size, facecolor="#E8EDF0", edgecolor=EDGE, linewidth=0.55, zorder=2))
    for row in range(5):
        for col in range(5):
            face = BLUE_MID if pattern[row][col] else WHITE
            ax.add_patch(
                Rectangle(
                    (x + col * cell, y + (4 - row) * cell),
                    cell,
                    cell,
                    facecolor=face,
                    edgecolor="#8AA1B1",
                    linewidth=0.35,
                    zorder=3,
                )
            )
    ax.add_patch(Rectangle((x, y), size, size, fill=False, edgecolor=EDGE, linewidth=0.8, zorder=4))


def draw_output_matrix(ax, x: float, y: float, w: float, h: float) -> None:
    cols, rows = 7, 5
    ax.add_patch(Rectangle((x + 0.7, y - 0.7), w, h, facecolor="#DDE5EA", edgecolor=EDGE, linewidth=0.55, zorder=1))
    for row in range(rows):
        for col in range(cols):
            level = (row * 2 + col) % 5
            colors = ["#F7FBFE", "#E8F2FA", "#D5E7F4", "#BED9EC", "#9CC3DF"]
            ax.add_patch(
                Rectangle(
                    (x + col * w / cols, y + row * h / rows),
                    w / cols,
                    h / rows,
                    facecolor=colors[level],
                    edgecolor="#8AA1B1",
                    linewidth=0.3,
                    zorder=3,
                )
            )
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=BLUE, linewidth=0.9, zorder=4))


def build_figure():
    fig = plt.figure(figsize=(183 / 25.4, 119 / 25.4), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 65)
    ax.axis("off")

    # Panel backgrounds and labels.
    for x, y, w, h in [(1.3, 45.2, 97.4, 18.2), (1.3, 1.5, 97.4, 41.7)]:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.35,rounding_size=1.2",
                facecolor=PANEL_BG,
                edgecolor=PANEL_EDGE,
                linewidth=0.85,
                zorder=0,
            )
        )
    ax.text(2.1, 62.0, "a", fontsize=9.2, fontweight="bold", color=INK, ha="left", va="top")
    ax.text(4.5, 62.0, "BISON full-band operator", fontsize=8.5, fontweight="bold", color=INK, ha="left", va="top")
    ax.text(2.1, 41.8, "b", fontsize=9.2, fontweight="bold", color=INK, ha="left", va="top")
    ax.text(4.5, 41.8, "One BandBlock: global spectral mixing, adaptive gating, and residual context", fontsize=8.5, fontweight="bold", color=INK, ha="left", va="top")

    # Panel a: macro architecture.
    draw_input_grid(ax, 4.0, 49.4, 7.2)
    ax.text(7.6, 48.0, r"permittivity grid  $\epsilon$", ha="center", va="top", fontsize=5.8, color=INK, fontweight="bold")
    ax.text(7.6, 46.5, r"$[B,1,H,W]$", ha="center", va="top", fontsize=5.2, color=EDGE)

    add_box(ax, 15.2, 50.0, 10.5, 8.5, r"$1\!\times\!1$ lift", "Conv2d  $1\\rightarrow32$\n$[B,32,H,W]$", face=BLUE_LIGHT)

    for idx in range(3, -1, -1):
        off = idx * 0.75
        ax.add_patch(
            FancyBboxPatch(
                (31.0 + off, 49.8 + off * 0.35),
                15.5,
                9.0,
                boxstyle="round,pad=0.22,rounding_size=0.9",
                facecolor=PURPLE_LIGHT if idx == 0 else "#E1E7EE",
                edgecolor=BLUE if idx == 0 else "#8AA1B1",
                linewidth=0.9 if idx == 0 else 0.55,
                zorder=4 - idx,
            )
        )
    ax.text(38.75, 55.9, "BandBlock", ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK, zorder=6)
    ax.text(38.75, 53.6, r"dual spectral + local + gate", ha="center", va="center", fontsize=5.3, color=EDGE, zorder=6)
    ax.text(38.75, 51.8, r"$[B,32,H,W]$", ha="center", va="center", fontsize=5.2, color=EDGE, zorder=6)
    ax.text(47.8, 59.1, r"$\times4$", ha="left", va="center", fontsize=6.3, color=BLUE, fontweight="bold")

    add_box(ax, 52.2, 50.0, 10.7, 8.5, "Global pool", "AdaptiveAvgPool2d(1)\n$[B,32]$", face=GRAY_LIGHT)
    add_box(ax, 67.2, 50.0, 11.0, 8.5, "Band head", "Linear $32\\rightarrow N_kN_b$\nreshape", face=ORANGE_LIGHT)
    draw_output_matrix(ax, 85.0, 50.0, 8.2, 8.5)
    ax.text(89.1, 48.0, r"full band matrix  $\widehat{\mathbf{W}}$", ha="center", va="top", fontsize=5.8, color=INK, fontweight="bold")
    ax.text(89.1, 46.5, r"$[B,N_k,N_b]$  (31$\times$10 evaluated)", ha="center", va="top", fontsize=5.0, color=EDGE)

    for start, end in [((11.8, 54.25), (14.7, 54.25)), ((26.0, 54.25), (30.3, 54.25)), ((49.4, 54.25), (51.5, 54.25)), ((63.2, 54.25), (66.5, 54.25)), ((78.5, 54.25), (84.3, 54.25))]:
        add_arrow(ax, start, end)
    ax.text(50.0, 60.7, "periodic field  →  global frequency mixing  →  jointly predicted dispersion", ha="center", va="center", fontsize=5.4, color=EDGE)

    # Panel b: detailed block.
    add_box(ax, 3.7, 23.4, 8.2, 8.0, r"input $X_\ell$", r"$[B,32,H,W]$", face=WHITE, title_size=7.0)
    ax.text(7.8, 21.7, "shared by all branches", ha="center", va="top", fontsize=4.8, color=EDGE)

    add_group(ax, 14.2, 30.1, 31.5, 9.8, "GLOBAL SPECTRAL PATH", BLUE_LIGHT)
    add_box(ax, 16.3, 32.2, 6.0, 5.2, "RFFT2", face=WHITE, title_size=6.5)
    add_box(ax, 25.0, 32.0, 11.8, 5.6, "complex mode mixing", "full $8\times8$ + band $5\times5$", face=WHITE, title_size=6.0, subtitle_size=4.9)
    add_box(ax, 39.4, 32.2, 4.7, 5.2, "iRFFT2", face=WHITE, title_size=6.1)
    add_arrow(ax, (22.5, 34.8), (24.5, 34.8), mutation=6.5)
    add_arrow(ax, (37.1, 34.8), (38.9, 34.8), mutation=6.5)

    add_group(ax, 14.2, 21.9, 24.0, 6.3, "LOCAL PATH", GRAY_LIGHT)
    add_box(ax, 18.4, 23.0, 15.5, 3.8, r"pointwise Conv $1\!\times\!1$", r"$32\rightarrow32$", face=WHITE, title_size=5.9, subtitle_size=4.8)

    add_group(ax, 14.2, 7.2, 26.8, 11.8, "LOW-RANK GLOBAL CONTEXT", GREEN_LIGHT)
    add_box(ax, 16.1, 9.3, 5.2, 5.2, "GAP", face=WHITE, title_size=6.4)
    add_box(ax, 23.1, 9.0, 15.2, 5.8, "channel bottleneck", "Linear $32\\rightarrow8$  •  ReLU\nLinear $8\\rightarrow32$  •  broadcast", face=WHITE, title_size=5.8, subtitle_size=4.7)
    add_arrow(ax, (21.5, 11.9), (22.6, 11.9), mutation=6.3)

    # Input split and identity shortcut.
    add_route(ax, [(12.0, 27.4), (13.2, 27.4), (13.2, 34.8), (15.8, 34.8)])
    add_route(ax, [(13.2, 27.4), (13.2, 24.9), (17.9, 24.9)])
    add_route(ax, [(7.8, 23.2), (7.8, 11.9), (15.6, 11.9)])
    add_route(ax, [(7.8, 31.7), (7.8, 39.7), (87.0, 39.7), (87.0, 29.0)], dashed=True)
    ax.text(66.5, 40.1, r"identity residual  $X_\ell$", ha="center", va="bottom", fontsize=5.2, color=EDGE)

    # Spectral and local merge.
    add_operator_node(ax, 49.2, 28.9, "+")
    add_route(ax, [(44.4, 34.8), (46.0, 34.8), (46.0, 29.8), (47.7, 29.2)])
    add_route(ax, [(38.6, 24.9), (45.3, 24.9), (45.3, 28.0), (47.7, 28.7)])
    ax.text(49.2, 26.9, r"$S_\ell$", ha="center", va="top", fontsize=5.4, color=EDGE, fontweight="bold")

    # Gate: S_l drives both the value path and the learned gate.
    add_box(ax, 54.0, 33.2, 11.0, 5.8, "Band Gate", r"Conv $1\!\times\!1$  →  Sigmoid", face=ORANGE_LIGHT, title_size=6.3, subtitle_size=4.8)
    add_operator_node(ax, 68.7, 28.9, r"$\times$", face="#5C9CC8")
    add_arrow(ax, (50.7, 28.9), (67.1, 28.9))
    add_route(ax, [(51.0, 29.7), (52.5, 29.7), (52.5, 36.1), (53.5, 36.1)])
    add_route(ax, [(65.5, 36.1), (68.7, 36.1), (68.7, 30.5)])
    ax.text(59.5, 31.9, r"$G_\ell$", ha="center", va="top", fontsize=5.0, color=EDGE, fontweight="bold")
    ax.text(68.7, 26.9, r"$U_\ell$", ha="center", va="top", fontsize=5.4, color=EDGE, fontweight="bold")

    # Low-rank addition, learned projection, identity addition, activation.
    add_operator_node(ax, 74.0, 28.9, "+")
    add_arrow(ax, (70.3, 28.9), (72.4, 28.9), mutation=6.5)
    add_route(ax, [(38.7, 11.9), (41.7, 11.9), (41.7, 19.9), (74.0, 19.9), (74.0, 27.3)])
    ax.text(57.8, 20.5, r"$r_\ell\in\mathbb{R}^{B\times32\times1\times1}$ broadcast over $H\times W$", ha="center", va="bottom", fontsize=4.8, color=EDGE)
    add_box(ax, 77.0, 25.8, 7.0, 6.2, r"$1\!\times\!1$", "projection", face=BLUE_LIGHT, title_size=6.7, subtitle_size=4.9)
    add_arrow(ax, (75.6, 28.9), (76.5, 28.9), mutation=6.0)
    add_operator_node(ax, 87.0, 28.9, "+")
    add_arrow(ax, (84.5, 28.9), (85.4, 28.9), mutation=6.0)
    add_box(ax, 90.0, 25.9, 5.5, 6.0, "GELU", face=PURPLE_LIGHT, title_size=6.6)
    add_arrow(ax, (88.6, 28.9), (89.5, 28.9), mutation=6.0)
    add_arrow(ax, (95.8, 28.9), (98.0, 28.9), mutation=7.0)
    ax.text(97.4, 30.8, r"$X_{\ell+1}$", ha="right", va="bottom", fontsize=5.8, color=INK, fontweight="bold")

    # Compact equation strip.
    ax.add_patch(
        FancyBboxPatch(
            (43.0, 3.5),
            53.4,
            13.0,
            boxstyle="round,pad=0.25,rounding_size=0.8",
            facecolor=WHITE,
            edgecolor=PANEL_EDGE,
            linewidth=0.7,
            zorder=1,
        )
    )
    ax.text(44.2, 15.2, "BandBlock update", ha="left", va="top", fontsize=5.7, color=INK, fontweight="bold")
    left_eq = (
        r"$S_\ell=\mathrm{SpectralConv}(X_\ell)+\mathrm{Conv}_{1\times1}(X_\ell)$" "\n"
        r"$G_\ell=\sigma(\mathrm{Conv}_{1\times1}(S_\ell))$" "\n"
        r"$U_\ell=G_\ell\odot S_\ell$"
    )
    right_eq = (
        r"$r_\ell=\mathrm{MLP}_{32\rightarrow8\rightarrow32}(\mathrm{GAP}(X_\ell))$" "\n"
        r"$X_{\ell+1}=\mathrm{GELU}($" "\n"
        r"$\quad\mathrm{Conv}_{1\times1}(U_\ell+r_\ell)+X_\ell)$"
    )
    ax.text(44.3, 12.9, left_eq, ha="left", va="top", fontsize=5.4, color=INK, linespacing=1.45)
    ax.text(70.0, 12.9, right_eq, ha="left", va="top", fontsize=5.2, color=INK, linespacing=1.45)
    ax.text(95.4, 4.6, "data-driven surrogate; MPB remains the physical verifier", ha="right", va="bottom", fontsize=4.7, color=EDGE, fontstyle="italic")

    return fig


def main() -> None:
    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    svg_path = OUT_BASE.with_suffix(".svg")
    fig.savefig(svg_path, facecolor=WHITE)
    # Matplotlib places spaces after multiline SVG path commands. Removing
    # line-end whitespace keeps the editable asset clean under Git checks.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(OUT_BASE.with_suffix(".pdf"), facecolor=WHITE)
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=600, facecolor=WHITE)
    fig.savefig(
        OUT_BASE.with_suffix(".tiff"),
        dpi=600,
        facecolor=WHITE,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
