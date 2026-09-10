"""Render a registry-complete architecture atlas for phc2d neural models.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence


FORMATS = ("png", "svg", "pdf", "tiff")
PALETTE = {
    "ink": "#283044",
    "muted": "#626A7A",
    "line": "#4E5668",
    "panel": "#FAFAFC",
    "panel_edge": "#D8D9E2",
    "baseline": "#E0E0F0",
    "baseline_edge": "#676A9C",
    "operator": "#E0F0F0",
    "operator_edge": "#42949E",
    "local": "#F0E0D0",
    "local_edge": "#B47A48",
    "field": "#E4CCD8",
    "field_edge": "#9A6078",
    "inverse": "#E3EDD9",
    "inverse_edge": "#678650",
    "warning": "#F6E7C8",
    "warning_edge": "#A9792A",
    "hero": "#F0C0CC",
    "hero_edge": "#B84F69",
}


REGISTERED_MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    "Fixed full-band / pilot": (
        "mlp",
        "res_mlp",
        "res_mlp_wide",
        "pod_res_mlp",
        "lifted_cnn",
        "fno_lite",
        "fno2d_band",
        "fno2d_band_v2",
    ),
    "Continuous query": (
        "mlp_query",
        "deep_onet_query",
        "deep_onet_query_v2",
        "fno2d_query",
        "fno2d_query_v2",
    ),
    "TE field": (
        "unet_field",
        "unet_field_v2",
        "fno_field",
        "fno_field_v2",
    ),
    "Full-cell upgrades": (
        "mlp_mixer_band_v1",
        "res_mlp_gated_v2",
        "convnext_band_v1",
        "tfno2d_band_v3",
        "dual_lane_band_operator_v1",
        "global_local_band_operator_v1",
    ),
    "Component-attribution identities": (
        "fno2d_band_component_v1",
        "tfno2d_band_component_v1",
        "convnext_band_component_v1",
        "dual_lane_band_component_v1",
    ),
}


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.facecolor"] = "white"


def _canvas(title: str, subtitle: str, figsize: tuple[float, float]) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=figsize)
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.set_axis_off()
    axis.text(2, 98, title, va="top", fontsize=18, fontweight="bold", color=PALETTE["ink"])
    axis.text(2, 93.7, subtitle, va="top", fontsize=9.5, color=PALETTE["muted"])
    return figure, axis


def _panel(axis: Any, x: float, y: float, width: float, height: float, title: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.35,rounding_size=1.0",
            facecolor=PALETTE["panel"],
            edgecolor=PALETTE["panel_edge"],
            linewidth=1.0,
        )
    )
    axis.text(x + 1.3, y + height - 1.6, title, va="top", fontsize=10.2, fontweight="bold", color=PALETTE["ink"])


def _box(
    axis: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    detail: str = "",
    *,
    kind: str = "baseline",
    fontsize: float = 8.3,
    dashed: bool = False,
) -> None:
    from matplotlib.patches import FancyBboxPatch, Polygon

    edge = f"{kind}_edge"
    offset = 0.38
    axis.add_patch(
        Polygon(
            (
                (x + width, y + 0.35),
                (x + width + offset, y),
                (x + width + offset, y + height - offset),
                (x + width, y + height),
            ),
            closed=True,
            facecolor=PALETTE[edge],
            edgecolor="none",
            alpha=0.20,
            zorder=1,
        )
    )
    axis.add_patch(
        Polygon(
            (
                (x + 0.35, y),
                (x + width + offset, y),
                (x + width, y + 0.35),
                (x, y + 0.35),
            ),
            closed=True,
            facecolor=PALETTE[edge],
            edgecolor="none",
            alpha=0.13,
            zorder=1,
        )
    )
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.25,rounding_size=0.65",
            facecolor=PALETTE[kind],
            edgecolor=PALETTE[edge],
            linewidth=1.1,
            linestyle="--" if dashed else "-",
            zorder=2,
        )
    )
    center = y + height / 2 + (0.9 if detail else 0)
    axis.text(x + width / 2, center, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=PALETTE["ink"], zorder=3)
    if detail:
        axis.text(
            x + width / 2,
            center - 2.1,
            detail,
            ha="center",
            va="top",
            fontsize=max(6.2, fontsize - 1.4),
            linespacing=1.18,
            color=PALETTE["muted"],
            zorder=3,
        )


def _arrow(
    axis: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "line",
    dashed: bool = False,
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": PALETTE[color],
            "linewidth": 1.35,
            "linestyle": "--" if dashed else "-",
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def _flow(
    axis: Any,
    y: float,
    items: Sequence[tuple[str, str, str]],
    *,
    x0: float,
    x1: float,
    height: float = 8.5,
) -> None:
    gap = 2.0
    width = (x1 - x0 - gap * (len(items) - 1)) / len(items)
    for index, (label, detail, kind) in enumerate(items):
        x = x0 + index * (width + gap)
        _box(axis, x, y, width, height, label, detail, kind=kind, fontsize=7.8)
        if index < len(items) - 1:
            _arrow(axis, (x + width, y + height / 2), (x + width + gap, y + height / 2))


def _tensor_stack(axis: Any, x: float, y: float, width: float, height: float, label: str, detail: str, *, kind: str) -> None:
    from matplotlib.patches import Rectangle

    for offset, alpha in ((0.9, 0.35), (0.45, 0.58), (0.0, 0.88)):
        axis.add_patch(Rectangle((x + offset, y + offset), width, height, facecolor=PALETTE[kind], edgecolor=PALETTE[f"{kind}_edge"], linewidth=1.0, alpha=alpha))
    axis.text(x + width / 2, y + height / 2 + 0.7, label, ha="center", va="center", fontsize=7.8, fontweight="bold", color=PALETTE["ink"])
    axis.text(x + width / 2, y + height / 2 - 1.3, detail, ha="center", va="center", fontsize=6.4, color=PALETTE["muted"])


def _registry_specs() -> dict[str, Any]:
    from phc2d.models import list_model_specs

    return {
        spec.name: spec
        for spec in list_model_specs()
        if getattr(spec, "alias_of", None) is None
    }


def _shape_text(value: Any) -> str:
    if value is None:
        return "dynamic"
    if isinstance(value, (tuple, list)):
        return "x".join(str(item) for item in value)
    return str(value)


def registry_coverage() -> dict[str, Any]:
    """Return the live registry-versus-atlas coverage contract."""

    registered = set(_registry_specs())
    declared = {name for names in REGISTERED_MODEL_GROUPS.values() for name in names}
    return {
        "registered": sorted(registered),
        "declared": sorted(declared),
        "missing": sorted(registered - declared),
        "extra": sorted(declared - registered),
    }


def _plot_architecture_poster(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        "phc2d architecture poster: 27 registered models + inverse generators",
        "Paper-style map of the main computational motifs; each family is expanded in the following atlas figures.",
        (19.0, 12.5),
    )
    cards = (
        (2, 52, "1 | Fixed Full-Wh (8)", "baseline", (("vector", "MLP / ResMLP", "baseline"), ("local", "Lifted CNN", "local"), ("spectral", "FNO / BandBlock", "operator"), ("Wh", "fixed matrix", "inverse"))),
        (35, 52, "2 | Continuous query (5)", "operator", (("structure", "grid encoder", "baseline"), ("k query", "MLP / trunk", "local"), ("fusion", "concat / product / gate", "operator"), ("bands", "at queried k", "inverse"))),
        (68, 52, "3 | TE field (4)", "field", (("grid + context", "periodic input", "field"), ("U-Net / FNO", "spatial backbone", "operator"), ("field map", "preserve H,W", "field"), ("frequency", "scalar head", "inverse"))),
        (2, 10, "4 | Full-cell upgrades (5)", "local", (("tokens", "Mixer", "baseline"), ("large kernel", "ConvNeXt", "local"), ("global op.", "TFNO", "operator"), ("lane FiLM", "conditional", "warning"))),
        (35, 10, "5 | Global-local bridge (1)", "hero", (("Tucker FFT", "global branch", "operator"), ("DWConv 7x7", "periodic local branch", "local"), ("fixed mean", "equal fusion", "hero"), ("Wh[81,10]", "shared head", "inverse"))),
        (68, 10, "6 | Attribution + inverse", "warning", (("frozen masks", "4 registry identities", "warning"), ("target Wh", "condition", "baseline"), ("mode + latent", "generator", "operator"), ("candidates", "analytic decode", "inverse"))),
    )
    for x, y, title, accent, items in cards:
        _panel(axis, x, y, 30, 36, title)
        axis.plot([x + 2, x + 28], [y + 30.8, y + 30.8], color=PALETTE[f"{accent}_edge"], linewidth=3.2, solid_capstyle="round")
        for index, (label, detail, kind) in enumerate(items):
            box_y = y + 23.5 - index * 6.0
            _box(axis, x + 4.2, box_y, 21.6, 4.8, label, detail, kind=kind, fontsize=7.0)
            if index < len(items) - 1:
                _arrow(axis, (x + 15, box_y), (x + 15, box_y - 1.2), color=f"{accent}_edge")
    axis.text(50, 4.4, "Color key: violet = vector/baseline | cyan = global operator | peach = local mixing | rose = fields/latest bridge | green = outputs", ha="center", fontsize=8.0, color=PALETTE["muted"])
    return _save_figure(figure, output_base, formats)


def _plot_registry_overview(output_base: Path, formats: Sequence[str]) -> list[Path]:
    specs = _registry_specs()
    figure, axis = _canvas(
        "phc2d model architecture atlas: live registry overview",
        "27 registered identities grouped by prediction contract. Component-attribution identities are models; inverse recipes and adapted PINO are shown separately.",
        (20.0, 12.5),
    )
    layouts = {
        "Fixed full-band / pilot": (2, 49, 31, 41),
        "Continuous query": (35, 49, 30, 41),
        "TE field": (67, 49, 31, 41),
        "Full-cell upgrades": (2, 7, 47, 37),
        "Component-attribution identities": (51, 7, 47, 37),
    }
    kinds = {
        "Fixed full-band / pilot": "baseline",
        "Continuous query": "operator",
        "TE field": "field",
        "Full-cell upgrades": "local",
        "Component-attribution identities": "warning",
    }
    for group, names in REGISTERED_MODEL_GROUPS.items():
        x, y, width, height = layouts[group]
        _panel(axis, x, y, width, height, f"{group} ({len(names)})")
        columns = 2 if len(names) >= 6 else 1
        rows = (len(names) + columns - 1) // columns
        card_width = (width - 4.4 - (columns - 1) * 1.4) / columns
        card_height = min(6.7, (height - 6.5 - (rows - 1) * 1.0) / rows)
        for index, name in enumerate(names):
            row, column = divmod(index, columns)
            card_x = x + 2.0 + column * (card_width + 1.4)
            card_y = y + height - 6.3 - (row + 1) * card_height - row * 1.0
            spec = specs[name]
            shape = _shape_text(spec.defaults.get("output_shape"))
            detail = f"{spec.input_repr} -> {spec.output_target} [{shape}]"
            _box(
                axis,
                card_x,
                card_y,
                card_width,
                card_height,
                name,
                detail,
                kind="hero" if name == "global_local_band_operator_v1" else kinds[group],
                fontsize=7.0 if len(name) > 26 else 7.7,
                dashed=name == "pod_res_mlp",
            )
    axis.text(
        50,
        2.2,
        "Dashed = registered placeholder. Pink = latest EXP-FWD-020 architecture; implemented, not authorized for training, and has no performance evidence.",
        ha="center",
        fontsize=8.2,
        color=PALETTE["muted"],
    )
    return _save_figure(figure, output_base, formats)


def _plot_forward_models(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        "Fixed full-band models: vector, local-convolution and spectral families",
        "The panel shows computational structure; v1/v2 names retain their own registry defaults and checkpoint identity.",
        (18.5, 11.0),
    )
    _panel(axis, 2, 61, 96, 29, "A | Vector MLP family")
    _flow(
        axis,
        72,
        (
            ("rho / grid", "flatten to d_in", "baseline"),
            ("input projection", "d_in -> hidden -> GELU", "baseline"),
            ("block x depth", "plain or residual 2-layer MLP", "operator"),
            ("linear head", "hidden -> Nk x Nb", "baseline"),
            ("Wh", "reshape [B,Nk,Nb]", "inverse"),
        ),
        x0=5,
        x1=95,
    )
    axis.text(6, 66.7, "mlp: no skip | res_mlp: 32-wide, depth 2 | res_mlp_wide: 256-wide, depth 6", fontsize=8.2, color=PALETTE["muted"])
    axis.text(6, 63.5, "pod_res_mlp: registry placeholder only; native POD projection/basis contract is not implemented", fontsize=8.2, color=PALETTE["warning_edge"], fontweight="bold")

    _panel(axis, 2, 34, 43, 22, "B | lifted_cnn")
    _flow(
        axis,
        42,
        (
            ("grid", "[B,C,H,W]", "baseline"),
            ("2 x Conv 3x3", "shared local filters + GELU", "local"),
            ("global average", "H,W -> 1,1", "operator"),
            ("linear head", "channels -> Wh", "inverse"),
        ),
        x0=5,
        x1=42,
        height=8.0,
    )
    axis.text(23.5, 36.5, "Zero padding; deliberately small local-feature baseline", ha="center", fontsize=7.7, color=PALETTE["muted"])

    _panel(axis, 48, 34, 50, 22, "C | fno_lite")
    _flow(
        axis,
        42,
        (
            ("grid", "[B,C,H,W]", "baseline"),
            ("1x1 lift", "C -> width", "baseline"),
            ("FNO block x d", "spectral + pointwise", "operator"),
            ("GAP + linear", "latent -> Wh", "inverse"),
        ),
        x0=51,
        x1=95,
        height=8.0,
    )
    axis.text(73, 36.5, "Minimal one-sided low-mode spectral baseline", ha="center", fontsize=7.7, color=PALETTE["muted"])

    _panel(axis, 2, 7, 96, 22, "D | fno2d_band, fno2d_band_v2 and fno2d_band_component_v1")
    _flow(
        axis,
        15,
        (
            ("grid", "epsilon / material weight", "baseline"),
            ("1x1 lift", "C -> width", "baseline"),
            ("BandBlock x d", "full + band Fourier weights", "operator"),
            ("gate + low-rank", "1x1 sigmoid; GAP MLP", "local"),
            ("skip + GELU", "residual feature update", "operator"),
            ("GAP + head", "linear or shared band head", "inverse"),
        ),
        x0=5,
        x1=95,
        height=8.2,
    )
    axis.text(50, 10.0, "Pilot: width 12 / depth 2 | formal v2: width 32 / depth 4 | component identity: same v2 backbone plus frozen component masks and shared dense head", ha="center", fontsize=7.7, color=PALETTE["muted"])
    return _save_figure(figure, output_base, formats)


def _plot_query_models(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        "Continuous-query models: three ways to combine structure and k",
        "All five registered identities predict a band vector at an explicit query coordinate; v2 changes capacity and formal data contract, not the fusion principle.",
        (18.5, 10.0),
    )
    _panel(axis, 2, 40, 30, 49, "A | mlp_query")
    _box(axis, 6, 72, 10, 7, "grid", "flatten", kind="baseline")
    _box(axis, 18, 72, 10, 7, "k", "query coordinate", kind="local")
    _arrow(axis, (11, 72), (13.5, 66))
    _arrow(axis, (23, 72), (16.5, 66), color="local_edge")
    _box(axis, 7, 59, 16, 7, "concatenate", "[flatten(grid), k]", kind="operator")
    _arrow(axis, (15, 59), (15, 55))
    _box(axis, 7, 48, 16, 7, "MLP x depth", "hidden layers -> band vector", kind="inverse")

    _panel(axis, 35, 40, 30, 49, "B | deep_onet_query / v2")
    _box(axis, 38, 70, 11, 9, "branch MLP", "flattened structure -> latent", kind="baseline")
    _box(axis, 52, 70, 10, 9, "trunk MLP", "k coordinate -> latent", kind="local")
    _arrow(axis, (43.5, 70), (48, 64))
    _arrow(axis, (57, 70), (52, 64), color="local_edge")
    _box(axis, 43, 57, 14, 7, "Hadamard product", "branch x trunk", kind="operator")
    _arrow(axis, (50, 57), (50, 53))
    _box(axis, 43, 46, 14, 7, "band head + bias", "latent -> [B,Nb]", kind="inverse")

    _panel(axis, 68, 40, 30, 49, "C | fno2d_query / v2")
    _box(axis, 71, 70, 11, 9, "FNO encoder", "grid -> spectral features -> GAP z", kind="operator")
    _box(axis, 85, 70, 10, 9, "query MLP", "k -> sigmoid gate", kind="local")
    _arrow(axis, (76.5, 70), (81, 64))
    _arrow(axis, (90, 70), (84, 64), color="local_edge")
    _box(axis, 75, 57, 15, 7, "gated fusion", "concat [z, z * gate]", kind="operator")
    _arrow(axis, (82.5, 57), (82.5, 53))
    _box(axis, 75, 46, 15, 7, "MLP head", "2 width -> [B,Nb]", kind="inverse")
    _panel(axis, 2, 7, 96, 27, "D | Registry-default variants")
    rows = (
        ("mlp_query", "12x14", "hidden 64; depth 2", "direct concatenation"),
        ("deep_onet_query", "12x14", "latent 32; depth 2", "branch-trunk product"),
        ("deep_onet_query_v2", "16x16", "latent 128; depth 4", "formal V2 identity"),
        ("fno2d_query", "12x14", "width 16; depth 2; modes 6x7", "query gate"),
        ("fno2d_query_v2", "16x16", "width 32; depth 4; modes 8x8", "formal V2 identity"),
    )
    axis.text(6, 28.4, "model", fontweight="bold", color=PALETTE["ink"])
    axis.text(35, 28.4, "input grid", fontweight="bold", color=PALETTE["ink"])
    axis.text(51, 28.4, "default capacity", fontweight="bold", color=PALETTE["ink"])
    axis.text(79, 28.4, "fusion", fontweight="bold", color=PALETTE["ink"])
    for index, row in enumerate(rows):
        y = 24.1 - index * 3.4
        for x, value in zip((6, 35, 51, 79), row, strict=True):
            axis.text(x, y, value, fontsize=7.7, color=PALETTE["muted"])
    return _save_figure(figure, output_base, formats)


def _plot_field_models(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        "TE eigenfield models: spatial output plus scalar frequency",
        "Legacy and v2 U-Net/FNO identities share the two-head task; v2 adds periodic handling, residual blocks and explicit field-context FiLM.",
        (18.5, 10.5),
    )
    _panel(axis, 2, 45, 46, 44, "A | unet_field and unet_field_v2")
    _flow(
        axis,
        69,
        (
            ("grid + context", "4 ch legacy / 5 ch v2", "field"),
            ("encoder", "Conv blocks + downsample", "local"),
            ("bottleneck", "deep spatial latent", "operator"),
            ("decoder", "upsample + skip concat", "local"),
            ("field head", "1x1 Conv", "field"),
        ),
        x0=5,
        x1=45,
        height=8.0,
    )
    _arrow(axis, (25, 69), (25, 60), color="field_edge")
    _box(axis, 16, 52, 18, 7, "frequency head", "GAP bottleneck (+ context in v2) -> scalar", kind="inverse", fontsize=7.5)
    axis.text(25, 47.8, "v2: circular 3x3 Conv + GroupNorm + residual + FiLM in every block", ha="center", fontsize=7.5, color=PALETTE["muted"])

    _panel(axis, 52, 45, 46, 44, "B | fno_field and fno_field_v2")
    _flow(
        axis,
        69,
        (
            ("grid + context", "4 ch legacy / 5 ch v2", "field"),
            ("1x1 lift", "channels -> width", "baseline"),
            ("operator x d", "Fourier + pointwise", "operator"),
            ("field head", "preserve H,W", "field"),
        ),
        x0=55,
        x1=95,
        height=8.0,
    )
    _arrow(axis, (78.5, 69), (78.5, 60), color="field_edge")
    _box(axis, 69, 52, 19, 7, "frequency head", "GAP feature (+ context in v2) -> scalar", kind="inverse", fontsize=7.5)
    axis.text(75, 47.8, "v2: +/- first-axis modes + GroupNorm + FiLM + residual update", ha="center", fontsize=7.5, color=PALETTE["muted"])

    _panel(axis, 2, 7, 96, 31, "C | What is and is not a separate architecture")
    _box(axis, 6, 18, 20, 10, "unet_field", "legacy supervised U-Net; width 32, depth 2", kind="field")
    _box(axis, 29, 18, 20, 10, "unet_field_v2", "periodic conditioned residual U-Net; width 48, depth 3", kind="field")
    _box(axis, 52, 18, 20, 10, "fno_field", "legacy data-only FNO field model; width 32, depth 4", kind="operator")
    _box(axis, 75, 18, 19, 10, "fno_field_v2", "conditioned FNO backbone; width 48, depth 4", kind="operator")
    axis.text(50, 11.6, "adapted PINO reuses fno_field_v2 and changes the training loss/regime; it is not a fifth registered field architecture.", ha="center", fontsize=8.2, color=PALETTE["warning_edge"], fontweight="bold")
    return _save_figure(figure, output_base, formats)


def _plot_full_cell_upgrades(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        "Full-cell Full-Wh upgrades: five established encoders and one new bridge",
        "Each encoder produces a global latent followed by the shared dense or k-conditioned band head. The new global-local model is expanded in the next figure.",
        (19.0, 11.0),
    )
    cards = (
        (2, 58, 30, 31, "A | mlp_mixer_band_v1", (("adaptive pool", "16x16 tokens", "baseline"), ("token projection", "C -> 256 + position", "local"), ("Mixer x 8", "token MLP then channel MLP", "operator"), ("LN + token mean", "z [B,256]", "inverse"))),
        (35, 58, 30, 31, "B | res_mlp_gated_v2", (("flatten + Linear", "4096 -> 320", "baseline"), ("gated block x 8", "GELU(Wg u) * Wv u", "local"), ("scale + DropPath", "residual add", "operator"), ("LayerNorm", "z [B,320]", "inverse"))),
        (68, 58, 30, 31, "C | convnext_band_v1", (("patch stem", "4x4 / stride 4", "baseline"), ("stages 2/3/6", "48 -> 96 -> 192", "local"), ("periodic DW 7x7", "channel MLP + residual", "operator"), ("GAP + LayerNorm", "z [B,192]", "inverse"))),
        (2, 16, 30, 34, "D | tfno2d_band_v3", (("1x1 lift", "C -> 64", "baseline"), ("operator x 6", "Tucker spectral + local 3x3 + channel", "operator"), ("learned gates", "normalized sigmoid fusion", "local"), ("GN + GELU + mean", "z [B,64]", "inverse"))),
        (35, 16, 30, 34, "E | dual_lane_band_operator_v1", (("grid + lane id", "Embedding(2,16)", "warning"), ("operator x 6", "TFNO three branches", "operator"), ("per-block FiLM", "gamma,beta from lane", "local"), ("GN + GELU + mean", "z [B,64]", "inverse"))),
        (68, 16, 30, 34, "F | global_local_band_operator_v1", (("1x1 lift", "C -> 64", "baseline"), ("operator x 6", "Tucker spectral + periodic local", "hero"), ("fixed mean", "no learned branch gate", "local"), ("GN + GELU + mean", "z [B,64]", "inverse"))),
    )
    for x, y, width, height, title, items in cards:
        _panel(axis, x, y, width, height, title)
        for index, (label, detail, kind) in enumerate(items):
            box_y = y + height - 12.0 - index * 6.1
            _box(axis, x + 4, box_y, width - 8, 5.0, label, detail, kind=kind, fontsize=7.2)
            if index < len(items) - 1:
                _arrow(axis, (x + width / 2, box_y), (x + width / 2, box_y - 1.1))
    axis.text(50, 7.8, "All six default to material_weight_grid -> Wh[81,10]; only dual-lane accepts lane_id. Interface support for k_points is not unseen-k validation.", ha="center", fontsize=8.2, color=PALETTE["muted"])
    return _save_figure(figure, output_base, formats)


def _plot_global_local(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        "global_local_band_operator_v1: fixed fusion of global and local physics features",
        "EXP-FWD-020 architecture contract: width 64, six blocks, Tucker modes 12x12 (rank 0.75), periodic 7x7 depthwise local mixing, expansion 4.",
        (19.0, 10.5),
    )
    _panel(axis, 2, 67, 96, 22, "A | End-to-end model")
    _flow(
        axis,
        74,
        (
            ("material grid", "[B,1,H,W]", "baseline"),
            ("1x1 lift", "1 -> 64", "baseline"),
            ("GlobalLocalBlock x 6", "same H,W; increasing DropPath", "hero"),
            ("GN + GELU", "spatial mean", "operator"),
            ("band head", "dense -> Wh[81,10]", "inverse"),
        ),
        x0=5,
        x1=95,
        height=8.5,
    )

    _panel(axis, 2, 21, 96, 40, "B | One GlobalLocalBlock")
    _tensor_stack(axis, 5, 37, 9, 9, "h_l", "64 x H x W", kind="baseline")
    _arrow(axis, (15, 42), (19, 42))
    _box(axis, 19, 37, 10, 9, "GroupNorm", "8 groups", kind="baseline")
    _arrow(axis, (29, 42), (33, 50), color="operator_edge")
    _arrow(axis, (29, 42), (33, 33), color="local_edge")
    _box(axis, 33, 47, 7, 7, "rFFT2", "+/- modes", kind="operator", fontsize=7.0)
    _arrow(axis, (40, 50.5), (43, 50.5), color="operator_edge")
    _box(axis, 43, 47, 11, 7, "Tucker core", "rank 0.75", kind="operator", fontsize=7.0)
    _arrow(axis, (54, 50.5), (57, 50.5), color="operator_edge")
    _box(axis, 57, 47, 7, 7, "irFFT2", "global", kind="operator", fontsize=7.0)
    _box(axis, 33, 29, 7, 8, "periodic", "pad", kind="local", fontsize=7.0)
    _arrow(axis, (40, 33), (43, 33), color="local_edge")
    _box(axis, 43, 29, 9, 8, "DWConv", "7x7", kind="local", fontsize=7.0)
    _arrow(axis, (52, 33), (55, 33), color="local_edge")
    _box(axis, 55, 29, 9, 8, "channel MLP", "64:256:64", kind="local", fontsize=7.0)
    _arrow(axis, (64, 50.5), (70, 42), color="operator_edge")
    _arrow(axis, (64, 33), (70, 42), color="local_edge")
    axis.scatter([71], [42], s=900, facecolor=PALETTE["hero"], edgecolor=PALETTE["hero_edge"], linewidth=1.2, zorder=2)
    axis.text(71, 42, "Σ/2", ha="center", va="center", fontsize=8.2, fontweight="bold", color=PALETTE["ink"], zorder=3)
    _arrow(axis, (72.8, 42), (76, 42))
    _box(axis, 76, 38, 8, 8, "DropPath", "update", kind="hero", fontsize=7.0)
    _arrow(axis, (84, 42), (87, 42))
    axis.scatter([88], [42], s=650, facecolor="white", edgecolor=PALETTE["hero_edge"], linewidth=1.2, zorder=2)
    axis.text(88, 42, "+", ha="center", va="center", fontsize=11, fontweight="bold", color=PALETTE["hero_edge"], zorder=3)
    _arrow(axis, (89.5, 42), (92, 42))
    _box(axis, 92, 38, 5, 8, "GELU", "h_(l+1)", kind="inverse", fontsize=6.8)
    _arrow(axis, (9.5, 37), (9.5, 24), color="hero_edge")
    axis.annotate(
        "",
        xy=(88, 40.4),
        xytext=(9.5, 24),
        arrowprops={
            "arrowstyle": "-|>",
            "color": PALETTE["hero_edge"],
            "linewidth": 1.35,
            "connectionstyle": "angle,angleA=0,angleB=90,rad=0",
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )

    _panel(axis, 2, 7, 96, 10, "C | Controlled identity and evidence boundary")
    axis.text(5, 11.7, "Compared with tfno2d_band_v3: remove the channel branch, replace learned sigmoid fusion with fixed mean, and upgrade local 3x3 dense Conv to periodic ConvNeXt-style 7x7 depthwise mixing.", fontsize=7.9, color=PALETTE["muted"])
    axis.text(5, 8.7, "Current status: implemented_not_authorized. No training, validation, held-out test, cross-domain, real-MPB or physical evidence exists yet.", fontsize=8.2, color=PALETTE["hero_edge"], fontweight="bold")
    return _save_figure(figure, output_base, formats)


def _plot_components_inverse(output_base: Path, formats: Sequence[str]) -> list[Path]:
    figure, axis = _canvas(
        "Component-attribution identities and implemented inverse generators",
        "Left: four registered EXP-FWD-019 model identities. Right: three implemented inverse architectures that are not part of the forward-model registry.",
        (19.0, 10.5),
    )
    _panel(axis, 2, 10, 49, 79, "A | Registered component-attribution models")
    component_rows = (
        ("fno2d_band_component_v1", "BandBlock backbone", "primary spectral | band spectral | gate | low-rank masks"),
        ("tfno2d_band_component_v1", "TFNO backbone", "spectral | local | channel branch masks"),
        ("convnext_band_component_v1", "ConvNeXt backbone", "kernel 7->1, expansion, padding and hierarchy variants"),
        ("dual_lane_band_component_v1", "lane-FiLM operator", "spectral | local | channel branch masks with lane id"),
    )
    for index, (name, backbone, mask) in enumerate(component_rows):
        y = 69 - index * 15.2
        _box(axis, 6, y, 21, 9.5, name, backbone, kind="warning", fontsize=7.1)
        _arrow(axis, (27, y + 4.8), (31, y + 4.8), color="warning_edge")
        _box(axis, 31, y, 16, 9.5, "frozen attribution control", mask, kind="operator", fontsize=6.9)
    axis.text(26.5, 15.2, "These identities retain checkpoint/model provenance; validation-only component evidence is not held-out or physical validation.", ha="center", fontsize=7.5, color=PALETTE["muted"])

    _panel(axis, 54, 10, 44, 79, "B | Non-registry inverse architectures")
    inverse_rows = (
        ("direct_inverse_mlp", (("target Wh", "condition", "baseline"), ("MLP", "2 hidden GELU layers", "operator"), ("sigmoid", "one structure", "inverse"))),
        ("single_mode_generator", (("Wh + z", "condition + Gaussian latent", "baseline"), ("conditional MLP", "2 hidden GELU layers", "operator"), ("sigmoid", "candidate set", "inverse"))),
        ("mixture_conditional_generator", (("Wh", "condition encoder", "baseline"), ("mode + z", "embedding + latent", "local"), ("deep decoder", "x4 GELU -> sigmoid", "operator"), ("candidates", "one-to-many", "inverse"))),
    )
    for row_index, (name, items) in enumerate(inverse_rows):
        y = 68 - row_index * 20.0
        axis.text(58, y + 9.5, name, fontsize=8.6, fontweight="bold", color=PALETTE["ink"])
        _flow(axis, y, items, x0=58, x1=94, height=7.7)
    axis.text(76, 15.2, "adapted PINO = fno_field_v2 backbone + physics-informed training losses; it is a regime, not another architecture.", ha="center", fontsize=7.5, color=PALETTE["warning_edge"], fontweight="bold")
    return _save_figure(figure, output_base, formats)


FIGURES: tuple[tuple[str, Callable[[Path, Sequence[str]], list[Path]]], ...] = (
    ("figure_0_model_architecture_poster", _plot_architecture_poster),
    ("figure_1_model_registry_overview", _plot_registry_overview),
    ("figure_2_forward_band_models", _plot_forward_models),
    ("figure_3_query_models", _plot_query_models),
    ("figure_4_field_models", _plot_field_models),
    ("figure_5_full_cell_upgrades", _plot_full_cell_upgrades),
    ("figure_6_global_local_band_operator_v1", _plot_global_local),
    ("figure_7_component_and_inverse_models", _plot_components_inverse),
)


def _save_figure(figure: Any, output_base: Path, formats: Sequence[str]) -> list[Path]:
    import matplotlib.pyplot as plt

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in formats:
        path = output_base.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.08,
            "facecolor": "white",
        }
        if suffix == "svg":
            kwargs["metadata"] = {"Date": None, "Creator": "phc2d"}
        elif suffix == "pdf":
            kwargs["metadata"] = {"Creator": "phc2d"}
        elif suffix == "png":
            kwargs.update({"dpi": 300, "metadata": {"Software": "phc2d"}})
        elif suffix == "tiff":
            kwargs.update({"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})
        figure.savefig(path, **kwargs)
        if suffix == "svg":
            normalized = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized + "\n", encoding="utf-8", newline="\n")
        paths.append(path)
    plt.close(figure)
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model_architecture_atlas(
    *, output_dir: Path, formats: Sequence[str] = FORMATS
) -> dict[str, Any]:
    """Render the atlas and write a machine-readable coverage manifest."""

    invalid = sorted(set(formats) - set(FORMATS))
    if invalid:
        raise ValueError(f"Unsupported model-atlas figure formats: {invalid}")
    coverage = registry_coverage()
    if coverage["missing"] or coverage["extra"]:
        raise RuntimeError(
            f"model atlas registry coverage mismatch: missing={coverage['missing']}, "
            f"extra={coverage['extra']}"
        )
    _style()
    destination = Path(output_dir)
    generated: list[Path] = []
    for stem, builder in FIGURES:
        generated.extend(builder(destination / stem, tuple(formats)))
    manifest = {
        "schema": "phc2d.model_architecture_atlas.v1",
        "status": "architecture_only_no_performance_claim",
        "registered_model_count": len(coverage["registered"]),
        "registered_models": coverage["registered"],
        "non_registry_architectures_shown": [
            "direct_inverse_mlp",
            "single_mode_generator",
            "mixture_conditional_generator",
        ],
        "training_regime_not_counted_as_architecture": ["adapted_pino"],
        "figure_count": len(FIGURES),
        "artifacts": [
            {"path": path.name, "sha256": _sha256(path)} for path in generated
        ],
    }
    manifest_path = destination / "atlas_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": "model_architecture_atlas_completed",
        "figure_count": len(FIGURES),
        "registered_model_count": len(coverage["registered"]),
        "artifact_count": len(generated) + 1,
        "artifacts": [path.as_posix() for path in generated] + [manifest_path.as_posix()],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=list(FORMATS), choices=FORMATS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_model_architecture_atlas(
        output_dir=args.output_dir,
        formats=tuple(args.formats),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
