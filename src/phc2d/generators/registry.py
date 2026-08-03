"""Structure generator abstractions for wedge-based dataset creation.

All generators output a ``StructureSample`` containing both the 36-entry wedge
vector and a 16×16 binary grid representation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from phc2d.data.lift import lift_rho_to_grid, wedge_indices


@dataclass(frozen=True)
class StructureSample:
    """A generated structure sample."""

    rho: np.ndarray
    binary_grid: np.ndarray
    metadata: dict[str, Any]
    postprocess_steps: list[str]


@dataclass(frozen=True)
class _GeneratorBase:
    """Base class for structure generators."""

    name: str

    def sample(self, *, rng: np.random.Generator, dataset_cfg: Mapping[str, Any]) -> StructureSample:
        raise NotImplementedError


def _wedge_rho_from_grid(binary_grid: np.ndarray) -> np.ndarray:
    """Extract a 36-entry wedge vector from a 16×16 binary grid."""

    wedge = wedge_indices(tuple(int(x) for x in binary_grid.shape))
    rho = np.zeros(len(wedge), dtype=np.float32)
    for idx, (i, j) in enumerate(wedge):
        rho[idx] = 1.0 if float(binary_grid[i, j]) >= 0.5 else 0.0
    return rho


def _resolve_fill_fraction(
    dataset_cfg: Mapping[str, Any],
    *,
    default_low: float,
    default_high: float,
) -> tuple[float, float]:
    """Resolve generator-specific fill-rate target."""

    target = dataset_cfg.get("fill_fraction", {})
    low = float(target.get("min", default_low)) if isinstance(target, Mapping) else float(default_low)
    high = float(target.get("max", default_high)) if isinstance(target, Mapping) else float(default_high)
    if low > high:
        low, high = high, low
    return max(0.0, min(1.0, low)), max(0.0, min(1.0, high))


def _symmetry_error(binary_grid: np.ndarray, symmetry: str) -> float:
    """Mean absolute difference from explicit symmetry projection."""

    if symmetry in ("", "none"):
        return 0.0
    projected = _apply_symmetry_projection(binary_grid, symmetry=symmetry)
    return float(np.mean(np.abs(binary_grid.astype(np.float32) - projected)))


def _apply_symmetry_projection(binary_grid: np.ndarray, *, symmetry: str) -> np.ndarray:
    """Project one mask to selected symmetry family by averaging group variants."""

    g = binary_grid.astype(np.float32)
    if symmetry in ("", "none"):
        return g

    if symmetry in ("c2", "mirror", "c2_or_mirror"):
        transforms = [
            g,
            np.rot90(g, 2),
        ]
    elif symmetry in ("c4", "dihedral"):
        transforms = [
            g,
            np.rot90(g, 1),
            np.rot90(g, 2),
            np.rot90(g, 3),
            np.fliplr(g),
            np.flipud(g),
            np.transpose(g),
            np.flipud(np.rot90(g, 1)),
        ]
    else:
        transforms = [g]
    return np.mean(np.stack(transforms, axis=0), axis=0)


def _binary_stats(binary_grid: np.ndarray) -> dict[str, float]:
    """Return compact binary shape statistics used by all generators."""

    g = (binary_grid > 0.5).astype(np.uint8)
    fill = float(g.mean())

    from scipy.ndimage import label, generate_binary_structure, binary_fill_holes

    structure = generate_binary_structure(2, 1)
    labeled, n = label(g, structure=structure)
    if n > 0:
        sizes = [int(np.sum(labeled == i)) for i in range(1, n + 1)]
        largest = max(sizes)
        largest_ratio = float(largest / max(g.sum(), 1))
    else:
        sizes = []
        largest_ratio = 0.0

    right = np.abs(g[:, 1:] - g[:, :-1]).sum()
    down = np.abs(g[1:, :] - g[:-1, :]).sum()
    perimeter = float(right + down)
    perimeter_area_ratio = perimeter / max(float(g.sum()), 1.0)

    filled = binary_fill_holes(g).astype(np.uint8)
    holes = int(np.sum(filled) - np.sum(g))

    neigh = np.zeros_like(g, dtype=np.int32)
    neigh[1:, :] += g[:-1, :]
    neigh[:-1, :] += g[1:, :]
    neigh[:, 1:] += g[:, :-1]
    neigh[:, :-1] += g[:, 1:]
    isolated = int(np.sum((g == 1) & (neigh == 0))) if g.sum() > 0 else 0

    min_feature_size = float(min(sizes)) if sizes else 0.0

    f = np.fft.fftshift(np.fft.fft2(g.astype(np.float32) - fill))
    pwr = np.abs(f) ** 2
    yy, xx = np.ogrid[: g.shape[0], : g.shape[1]]
    cy = (g.shape[0] - 1) / 2.0
    cx = (g.shape[1] - 1) / 2.0
    rr = (yy - cy) ** 2 + (xx - cx) ** 2
    low_mask = rr <= (4.0 ** 2)
    low_freq_ratio = float(np.sum(pwr[low_mask]) / max(np.sum(pwr), 1e-12))

    edge_cells = perimeter + 1.0
    total_pairs = float(g.size)
    edge_density = edge_cells / max(total_pairs, 1.0)

    return {
        "fill_fraction": fill,
        "num_connected_components": int(n),
        "largest_component_ratio": float(largest_ratio),
        "perimeter_area_ratio": float(perimeter_area_ratio),
        "min_feature_size": float(min_feature_size),
        "num_holes": int(holes),
        "isolated_pixel_count": int(isolated),
        "low_frequency_energy_ratio": float(low_freq_ratio),
        "edge_density": float(edge_density),
    }


def _randomize_symmetry(rng: np.random.Generator, candidate: tuple[str, ...]) -> str:
    if not candidate:
        return "none"
    idx = int(rng.integers(0, len(candidate)))
    return str(candidate[idx])


def _draw_primitive_raster(
    grid: np.ndarray,
    primitive: str,
    rng: np.random.Generator,
    *,
    invert: bool = False,
) -> None:
    """Draw one raster primitive into ``grid`` in-place."""

    n = grid.shape[0]
    yy, xx = np.ogrid[:n, :n]
    v = 1.0 - float(invert)

    if primitive == "rectangle":
        h = int(rng.integers(max(1, n // 12), max(2, n // 3)))
        w = int(rng.integers(max(1, n // 12), max(2, n // 3)))
        r0 = int(rng.integers(0, max(2, n - h)))
        c0 = int(rng.integers(0, max(2, n - w)))
        grid[r0 : r0 + h, c0 : c0 + w] = np.maximum(grid[r0 : r0 + h, c0 : c0 + w], v)
    elif primitive == "l_shape":
        h = int(rng.integers(max(1, n // 14), max(2, n // 4)))
        w = int(rng.integers(max(1, n // 14), max(2, n // 4)))
        thickness = max(1, int(max(1, n // 20)))
        r0 = int(rng.integers(0, max(2, n - h)))
        c0 = int(rng.integers(0, max(2, n - w)))
        grid[r0 : r0 + h, c0 : min(c0 + thickness, n)] = np.maximum(grid[r0 : r0 + h, c0 : min(c0 + thickness, n)], v)
        grid[r0 : min(r0 + thickness, n), c0 : c0 + w] = np.maximum(grid[r0 : min(r0 + thickness, n), c0 : c0 + w], v)
    elif primitive == "t_shape":
        h = int(rng.integers(max(1, n // 14), max(2, n // 4)))
        w = int(rng.integers(max(1, n // 14), max(2, n // 4)))
        t = max(1, int(max(1, n // 20)))
        r0 = int(rng.integers(0, max(2, n - h)))
        c0 = int(rng.integers(0, max(2, n - w)))
        hh = max(t, h // 3)
        grid[r0 : r0 + hh, c0 : c0 + w] = np.maximum(grid[r0 : r0 + hh, c0 : c0 + w], v)
        vc = c0 + w // 2
        grid[r0 : r0 + h, max(0, vc - t // 2) : min(n, vc + t // 2 + 1)] = np.maximum(
            grid[r0 : r0 + h, max(0, vc - t // 2) : min(n, vc + t // 2 + 1)],
            v,
        )
    elif primitive == "ring":
        r = int(rng.integers(max(1, n // 18), max(2, n // 4)))
        cy = int(rng.integers(r, max(r + 1, n - r)))
        cx = int(rng.integers(r, max(r + 1, n - r)))
        rr = (yy - cy) ** 2 + (xx - cx) ** 2
        thick = max(1, int(rng.integers(max(1, n // 20), max(2, n // 12))))
        outer = r + thick
        mask = (rr >= r * r) & (rr <= outer * outer)
        grid[mask] = np.maximum(grid[mask], v)
    elif primitive == "cross":
        r = int(rng.integers(max(2, n // 10), max(3, n // 4)))
        r0 = int(rng.integers(0, max(2, n - r)))
        c0 = int(rng.integers(0, max(2, n - r)))
        thickness = max(1, int(rng.integers(max(1, n // 20), max(2, n // 10)))
        )
        grid[r0 : r0 + r, max(0, c0 + r // 2 - thickness // 2) : min(n, c0 + r // 2 + thickness // 2 + 1)] = np.maximum(
            grid[r0 : r0 + r, max(0, c0 + r // 2 - thickness // 2) : min(n, c0 + r // 2 + thickness // 2 + 1)],
            v,
        )
        grid[max(0, r0 + r // 2 - thickness // 2) : min(n, r0 + r // 2 + thickness // 2 + 1), c0 : c0 + r] = np.maximum(
            grid[max(0, r0 + r // 2 - thickness // 2) : min(n, r0 + r // 2 + thickness // 2 + 1), c0 : c0 + r],
            v,
        )
    else:
        cx = float(rng.uniform(0.15, 0.85) * (n - 1))
        cy = float(rng.uniform(0.15, 0.85) * (n - 1))
        rad = float(rng.uniform(n / 18.0, n / 8.0))
        rr = (yy - cy) ** 2 + (xx - cx) ** 2
        grid[rr <= rad ** 2] = np.maximum(grid[rr <= rad ** 2], v)


class LegacyRandomGenerator(_GeneratorBase):
    """Legacy Bernoulli 36-bit wedge generator (default behavior)."""

    def __init__(self) -> None:
        super().__init__(name="legacy_random")

    def sample(self, *, rng: np.random.Generator, dataset_cfg: Mapping[str, Any]) -> StructureSample:
        prob = float(dataset_cfg.get("prob_high", 0.5))
        prob = float(max(0.0, min(1.0, prob)))
        n_wedge = int(dataset_cfg.get("n_wedge", 36))
        rho = (rng.random(n_wedge) < prob).astype(np.float32)
        grid = lift_rho_to_grid(rho)
        return StructureSample(
            rho=rho.astype(np.float32),
            binary_grid=grid,
            metadata={
                "symmetry": "d4_legacy",
            },
            postprocess_steps=["legacy_random"],
        )


def _normalized_weights(weights: Mapping[str, Any] | None, names: tuple[str, ...]) -> np.ndarray:
    if not isinstance(weights, Mapping):
        return np.ones(len(names), dtype=np.float64) / float(len(names))
    vals = np.asarray([max(0.0, float(weights.get(name, 0.0))) for name in names], dtype=np.float64)
    if float(vals.sum()) <= 0.0:
        return np.ones(len(names), dtype=np.float64) / float(len(names))
    return vals / float(vals.sum())


def _clip_span(start: int, end: int, n: int = 16) -> tuple[int, int]:
    return max(0, int(start)), min(int(end), int(n))


def _rect(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int, *, value: float) -> None:
    rr0, rr1 = _clip_span(r0, r1, grid.shape[0])
    cc0, cc1 = _clip_span(c0, c1, grid.shape[1])
    if rr0 < rr1 and cc0 < cc1:
        grid[rr0:rr1, cc0:cc1] = value


def _even_width(width: int, *, lower: int = 2, upper: int = 16) -> int:
    width = max(int(lower), min(int(upper), int(width)))
    if width % 2 == 1:
        width = width + 1 if width < upper else width - 1
    return max(int(lower), min(int(upper), int(width)))


def _centered_span(width: int, *, n: int = 16) -> tuple[int, int]:
    width = _even_width(width, lower=2, upper=n)
    start = (n - width) // 2
    return start, start + width


def _centered_square(grid: np.ndarray, width: int, *, value: float) -> tuple[int, int]:
    s, e = _centered_span(width, n=grid.shape[0])
    _rect(grid, s, e, s, e, value=value)
    return s, e


def _corner_pads(grid: np.ndarray, size: int, inset: int, *, value: float) -> None:
    size = _even_width(size, lower=2, upper=grid.shape[0] // 2)
    inset = max(0, min(int(inset), grid.shape[0] // 2 - size))
    n = grid.shape[0]
    positions = [
        (inset, inset),
        (inset, n - inset - size),
        (n - inset - size, inset),
        (n - inset - size, n - inset - size),
    ]
    for r0, c0 in positions:
        _rect(grid, r0, r0 + size, c0, c0 + size, value=value)


def _diagonal_satellites(grid: np.ndarray, size: int, offset: int, *, value: float) -> None:
    size = _even_width(size, lower=2, upper=grid.shape[0] // 3)
    offset = max(1, min(int(offset), 4))
    c = grid.shape[0] // 2
    positions = [
        (c - offset - size, c - offset - size),
        (c - offset - size, c + offset),
        (c + offset, c - offset - size),
        (c + offset, c + offset),
    ]
    for r0, c0 in positions:
        _rect(grid, r0, r0 + size, c0, c0 + size, value=value)


def _cardinal_bridges(grid: np.ndarray, inner_width: int, outer_width: int, bridge_width: int, *, value: float) -> None:
    n = grid.shape[0]
    inner_s, inner_e = _centered_span(inner_width, n=n)
    outer_s, outer_e = _centered_span(outer_width, n=n)
    bw = _even_width(bridge_width, lower=2, upper=4)
    bs, be = _centered_span(bw, n=n)
    _rect(grid, outer_s, inner_s, bs, be, value=value)
    _rect(grid, inner_e, outer_e, bs, be, value=value)
    _rect(grid, bs, be, outer_s, inner_s, value=value)
    _rect(grid, bs, be, inner_e, outer_e, value=value)


def _ring_frame(grid: np.ndarray, outer_width: int, thickness: int, *, value: float) -> tuple[int, int, int]:
    outer_width = _even_width(outer_width, lower=6, upper=14)
    thickness = _even_width(thickness, lower=2, upper=max(2, outer_width - 4))
    thickness = min(thickness, outer_width // 2 - 1)
    outer_s, outer_e = _centered_span(outer_width, n=grid.shape[0])
    _rect(grid, outer_s, outer_e, outer_s, outer_e, value=value)
    hole = outer_width - 2 * thickness
    if hole >= 2:
        inner_s, inner_e = _centered_span(hole, n=grid.shape[0])
        _rect(grid, inner_s, inner_e, inner_s, inner_e, value=0.0)
    return outer_s, outer_e, thickness


def _center_cross_void(grid: np.ndarray, cross_width: int, extent_width: int) -> None:
    n = grid.shape[0]
    cross_width = _even_width(cross_width, lower=2, upper=min(6, n))
    extent_width = _even_width(extent_width, lower=max(cross_width + 2, 6), upper=n)
    cs, ce = _centered_span(cross_width, n=n)
    es, ee = _centered_span(extent_width, n=n)
    _rect(grid, es, ee, cs, ce, value=0.0)
    _rect(grid, cs, ce, es, ee, value=0.0)


def _side_notches(grid: np.ndarray, outer_start: int, outer_end: int, thickness: int, notch_width: int, notch_depth: int) -> None:
    notch_width = _even_width(notch_width, lower=2, upper=max(2, outer_end - outer_start - 2))
    notch_depth = max(1, min(int(notch_depth), max(1, thickness)))
    ns, ne = _centered_span(notch_width, n=grid.shape[0])
    _rect(grid, outer_start, outer_start + notch_depth, ns, ne, value=0.0)
    _rect(grid, outer_end - notch_depth, outer_end, ns, ne, value=0.0)
    _rect(grid, ns, ne, outer_start, outer_start + notch_depth, value=0.0)
    _rect(grid, ns, ne, outer_end - notch_depth, outer_end, value=0.0)


def _mask_bbox(binary_grid: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(binary_grid > 0.5)
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _edge_windows(binary_grid: np.ndarray, window_width: int, window_depth: int) -> bool:
    bbox = _mask_bbox(binary_grid)
    if bbox is None:
        return False
    r0, r1, c0, c1 = bbox
    if (r1 - r0) < 6 or (c1 - c0) < 6:
        return False
    window_width = _even_width(window_width, lower=2, upper=min(6, c1 - c0 - 2))
    window_depth = max(1, min(int(window_depth), 2))
    rs, re = _centered_span(window_width, n=binary_grid.shape[0])
    cs, ce = _centered_span(window_width, n=binary_grid.shape[1])
    _rect(binary_grid, r0, r0 + window_depth, cs, ce, value=0.0)
    _rect(binary_grid, r1 - window_depth, r1, cs, ce, value=0.0)
    _rect(binary_grid, rs, re, c0, c0 + window_depth, value=0.0)
    _rect(binary_grid, rs, re, c1 - window_depth, c1, value=0.0)
    return True


def _cardinal_tabs(binary_grid: np.ndarray, tab_width: int, tab_length: int) -> bool:
    bbox = _mask_bbox(binary_grid)
    if bbox is None:
        return False
    r0, r1, c0, c1 = bbox
    tab_width = _even_width(tab_width, lower=2, upper=4)
    tab_length = _even_width(tab_length, lower=2, upper=6)
    rs, re = _centered_span(tab_width, n=binary_grid.shape[0])
    cs, ce = _centered_span(tab_width, n=binary_grid.shape[1])
    _rect(binary_grid, max(0, r0 - tab_length), r0, cs, ce, value=1.0)
    _rect(binary_grid, r1, min(binary_grid.shape[0], r1 + tab_length), cs, ce, value=1.0)
    _rect(binary_grid, rs, re, max(0, c0 - tab_length), c0, value=1.0)
    _rect(binary_grid, rs, re, c1, min(binary_grid.shape[1], c1 + tab_length), value=1.0)
    return True


def _remove_small_components(binary_grid: np.ndarray, min_area: int) -> np.ndarray:
    from scipy.ndimage import label, generate_binary_structure

    g = (binary_grid > 0.5).astype(np.uint8)
    if min_area <= 1:
        return g.astype(np.float32)
    labeled, n = label(g, structure=generate_binary_structure(2, 1))
    for comp in range(1, n + 1):
        mask = labeled == comp
        if int(mask.sum()) < int(min_area):
            g[mask] = 0
    return g.astype(np.float32)


def _fill_small_holes(binary_grid: np.ndarray, max_hole_area: int) -> np.ndarray:
    from scipy.ndimage import label, generate_binary_structure

    g = (binary_grid > 0.5).astype(np.uint8)
    if max_hole_area < 1:
        return g.astype(np.float32)
    inv = 1 - g
    labeled, n = label(inv, structure=generate_binary_structure(2, 1))
    for comp in range(1, n + 1):
        mask = labeled == comp
        rows, cols = np.where(mask)
        if rows.size == 0:
            continue
        touches_boundary = (
            np.any(rows == 0)
            or np.any(rows == g.shape[0] - 1)
            or np.any(cols == 0)
            or np.any(cols == g.shape[1] - 1)
        )
        if not touches_boundary and int(mask.sum()) <= int(max_hole_area):
            g[mask] = 1
    return g.astype(np.float32)


def _cleanup_paper_like_binary(binary_grid: np.ndarray, *, min_component_area: int, max_hole_area: int) -> np.ndarray:
    g = _remove_small_components(binary_grid, min_area=min_component_area)
    g = _fill_small_holes(g, max_hole_area=max_hole_area)
    return (g > 0.5).astype(np.float32)


def _apply_paper_like_morphology(
    binary_grid: np.ndarray,
    *,
    opening: bool,
    closing: bool,
    fill_holes: bool,
) -> np.ndarray:
    """Opening removes isolated pixels; closing and fill_holes seal small gaps."""

    from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, generate_binary_structure

    structure = generate_binary_structure(2, 1)
    g = binary_grid > 0.5
    if opening:
        g = binary_opening(g, structure=structure)
    if closing:
        g = binary_closing(g, structure=structure)
    if fill_holes:
        g = binary_fill_holes(g)
    return g.astype(np.float32)


def _limit_connected_components(binary_grid: np.ndarray, max_components: int) -> np.ndarray:
    """Keep only the largest connected components up to ``max_components``.

    When ``max_components <= 0``, the grid is returned unchanged.
    """

    if max_components <= 0:
        return (binary_grid > 0.5).astype(np.float32)

    from scipy.ndimage import generate_binary_structure, label

    g = (binary_grid > 0.5).astype(np.uint8)
    labeled, n = label(g, structure=generate_binary_structure(2, 1))
    if n <= max_components:
        return g.astype(np.float32)
    ranked = sorted(
        ((comp, int((labeled == comp).sum())) for comp in range(1, n + 1)),
        key=lambda item: item[1],
        reverse=True,
    )
    keep = {comp for comp, _ in ranked[:max_components]}
    out = np.zeros_like(g)
    for comp in keep:
        out[labeled == comp] = 1
    return out.astype(np.float32)


def _finalize_paper_like_binary(
    binary_grid: np.ndarray,
    *,
    min_area: int,
    max_components: int | None,
) -> np.ndarray:
    """Remove specks/isolated pixels while preserving separated blocks when allowed."""

    b = _remove_small_components(binary_grid, min_area=max(2, int(min_area)))
    if max_components is not None and int(max_components) > 0:
        b = _limit_connected_components(b, max_components=int(max_components))
    if int(_binary_stats(b)["isolated_pixel_count"]) > 0:
        b = _apply_paper_like_morphology(b, opening=True, closing=False, fill_holes=False)
        b = _remove_small_components(b, min_area=max(2, int(min_area)))
    return (b > 0.5).astype(np.float32)


def _apply_paper_like_variations(
    binary_grid: np.ndarray,
    rng: np.random.Generator,
    *,
    max_ops: int,
) -> tuple[np.ndarray, list[str]]:
    g = binary_grid.astype(np.float32).copy()
    ops_catalog = [
        "edge_windows",
        "cardinal_tabs",
        "diagonal_cutouts",
        "diagonal_satellites",
        "center_hole",
    ]
    n_ops = int(rng.integers(0, max(1, max_ops) + 1))
    chosen = list(rng.choice(np.array(ops_catalog, dtype=object), size=n_ops, replace=False)) if n_ops > 0 else []
    applied: list[str] = []
    for op in chosen:
        if op == "edge_windows":
            if _edge_windows(g, window_width=int(rng.choice([2, 4])), window_depth=int(rng.choice([1, 2]))):
                applied.append("edge_windows")
        elif op == "cardinal_tabs":
            if _cardinal_tabs(g, tab_width=int(rng.choice([2, 4])), tab_length=int(rng.choice([2, 4]))):
                applied.append("cardinal_tabs")
        elif op == "diagonal_cutouts":
            _diagonal_satellites(g, size=int(rng.choice([2, 4])), offset=int(rng.choice([1, 2, 3])), value=0.0)
            applied.append("diagonal_cutouts")
        elif op == "diagonal_satellites":
            _diagonal_satellites(g, size=int(rng.choice([2, 4])), offset=int(rng.choice([1, 2, 3])), value=1.0)
            applied.append("diagonal_satellites")
        elif op == "center_hole":
            bbox = _mask_bbox(g)
            if bbox is not None:
                r0, r1, c0, c1 = bbox
                if (r1 - r0) >= 6 and (c1 - c0) >= 6:
                    _centered_square(g, int(rng.choice([2, 4])), value=0.0)
                    applied.append("center_hole")
    return (g > 0.5).astype(np.float32), applied


def _random_primitive_count(rng: np.random.Generator, cfg: Mapping[str, Any]) -> int:
    lo = max(1, int(cfg.get("min_primitive_iters", 1)))
    hi = max(lo, int(cfg.get("max_primitive_iters", 16)))
    return int(rng.integers(lo, hi + 1))


def _boost_center_disk(binary_grid: np.ndarray, *, n: int = 16) -> np.ndarray:
    b = binary_grid.astype(np.float32).copy()
    yy, xx = np.ogrid[:n, :n]
    r = int(max(2, min(n // 6, n // 3)))
    center = (n / 2.0, n / 2.0)
    rr = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    b[rr <= r * r] = 1.0
    return b


def _process_paper_like_binary(
    binary_grid: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    min_area: int,
    max_components: int | None,
) -> np.ndarray:
    b = binary_grid.astype(np.float32)
    if bool(cfg.get("morphology_opening", True)):
        b = _apply_paper_like_morphology(b, opening=True, closing=False, fill_holes=False)
    if bool(cfg.get("morphology_closing", True)):
        b = _apply_paper_like_morphology(b, opening=False, closing=True, fill_holes=False)
    if bool(cfg.get("fill_holes", True)):
        b = _apply_paper_like_morphology(b, opening=False, closing=False, fill_holes=True)
    return _finalize_paper_like_binary(b, min_area=min_area, max_components=max_components)


def _paper_like_lifted_view(binary_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rho = _wedge_rho_from_grid(binary_grid)
    lifted = lift_rho_to_grid(rho).astype(np.float32)
    return rho, lifted


def _paper_like_is_valid_lifted(
    lifted: np.ndarray,
    *,
    min_fill: float,
    max_fill: float,
) -> bool:
    stats = _binary_stats(lifted)
    fill = float(stats["fill_fraction"])
    return (
        float(np.sum(lifted > 0.5)) > 0.0
        and min_fill <= fill <= max_fill
        and int(stats["isolated_pixel_count"]) == 0
    )


def _paper_like_force_nonempty(
    rng: np.random.Generator,
    cfg: Mapping[str, Any],
    *,
    min_fill: float,
    max_fill: float,
    symmetry_candidates: tuple[str, ...],
    min_area: int,
    max_components: int | None,
) -> tuple[np.ndarray, np.ndarray, str, list[str], list[str]]:
    """Build a guaranteed non-empty wedge-consistent sample with random block layout."""

    steps: list[str] = ["nonempty_fallback"]
    symmetry = str(_randomize_symmetry(rng, symmetry_candidates))
    n = 16
    g = np.zeros((n, n), dtype=np.float32)
    primitives_used: list[str] = []
    n_draws = _random_primitive_count(rng, cfg)
    for _ in range(n_draws):
        prim = str(rng.choice(["rectangle", "l_shape", "t_shape", "ring", "cross"]))
        _draw_primitive_raster(g, prim, rng, invert=False)
        primitives_used.append(prim)
        if float(g.mean()) > max_fill:
            break

    if symmetry != "none":
        g = _apply_symmetry_projection(g, symmetry=symmetry)
        steps.append(f"symmetry:{symmetry}")

    b = _process_paper_like_binary(
        (g >= 0.5).astype(np.float32),
        cfg,
        min_area=min_area,
        max_components=max_components,
    )
    if float(b.mean()) < min_fill:
        b = _boost_center_disk(b, n=n)
        b = _process_paper_like_binary(b, cfg, min_area=min_area, max_components=max_components)
        steps.append("fill_center_boost")

    rho, lifted = _paper_like_lifted_view(b)
    if not _paper_like_is_valid_lifted(lifted, min_fill=min_fill, max_fill=max_fill):
        lifted = np.zeros((n, n), dtype=np.float32)
        width = int(rng.choice(np.array([4, 6, 8], dtype=np.int64)))
        _centered_square(lifted, width, value=1.0)
        _corner_pads(lifted, int(rng.choice([2, 4])), int(rng.choice([1, 2, 3])), value=1.0)
        lifted = _process_paper_like_binary(lifted, cfg, min_area=min_area, max_components=max_components)
        rho = _wedge_rho_from_grid(lifted)
        lifted = lift_rho_to_grid(rho).astype(np.float32)
        steps.append("hard_center_fallback")

    return rho, lifted, symmetry, primitives_used, steps


def _paper_like_stats_ok(
    binary_grid: np.ndarray,
    *,
    min_fill: float,
    max_fill: float,
    max_components: int,
    min_low_freq_ratio: float,
) -> bool:
    stats = _binary_stats(binary_grid)
    if stats["fill_fraction"] < float(min_fill) or stats["fill_fraction"] > float(max_fill):
        return False
    if int(stats["isolated_pixel_count"]) != 0:
        return False
    if int(stats["num_connected_components"]) > int(max_components):
        return False
    if float(stats["low_frequency_energy_ratio"]) < float(min_low_freq_ratio):
        return False
    if float(_symmetry_error(binary_grid, "c4")) > 1.0e-6:
        return False
    if float(np.sum(binary_grid > 0.5)) <= 0.0:
        return False
    if float(np.sum(binary_grid > 0.5)) >= float(binary_grid.size):
        return False
    return True


class PaperLikeCompactGenerator(_GeneratorBase):
    """Compact connected masks built from low-frequency geometric primitives."""

    _PRIMITIVES = ("rectangle", "l_shape", "t_shape", "ring", "cross")

    def __init__(self, *, options: Mapping[str, Any] | None = None) -> None:
        super().__init__(name="paper_like_compact")
        self.options = dict(options or {})

    def _cfg(self, dataset_cfg: Mapping[str, Any]) -> dict[str, Any]:
        cfg = dict(dataset_cfg)
        cfg.update(self.options)
        cfg.setdefault("min_primitive_iters", 1)
        cfg.setdefault("max_primitive_iters", 16)
        cfg.setdefault("symmetry_candidates", ("c4", "c2", "mirror", "none"))
        cfg.setdefault("min_area", 2)
        cfg.setdefault("max_components", None)
        cfg.setdefault("morphology_opening", True)
        cfg.setdefault("morphology_closing", True)
        cfg.setdefault("fill_holes", True)
        return cfg

    def sample(self, *, rng: np.random.Generator, dataset_cfg: Mapping[str, Any]) -> StructureSample:
        cfg = self._cfg(dataset_cfg)
        n = 16
        min_fill, max_fill = _resolve_fill_fraction(cfg, default_low=0.15, default_high=0.75)
        symmetry_candidates = tuple(str(item) for item in cfg.get("symmetry_candidates", ("c4", "c2", "mirror", "none")))
        min_area = int(cfg.get("min_area", 2))
        max_components_raw = cfg.get("max_components", None)
        max_components = None if max_components_raw in (None, "null", "") else int(max_components_raw)

        steps: list[str] = ["paper_like_compact"]
        primitives_used: list[str] = []
        symmetry = "none"
        rho = np.zeros(36, dtype=np.float32)
        lifted = np.zeros((n, n), dtype=np.float32)

        for _ in range(32):
            symmetry = str(_randomize_symmetry(rng, symmetry_candidates))
            g = np.zeros((n, n), dtype=np.float32)
            primitives_used = []
            n_draws = _random_primitive_count(rng, cfg)
            for _prim in range(n_draws):
                prim = str(rng.choice(list(self._PRIMITIVES)))
                _draw_primitive_raster(g, prim, rng, invert=False)
                primitives_used.append(prim)
                if float(g.mean()) > max_fill:
                    break

            if symmetry != "none":
                g = _apply_symmetry_projection(g, symmetry=symmetry)

            b = _process_paper_like_binary(
                (g >= 0.5).astype(np.float32),
                cfg,
                min_area=min_area,
                max_components=max_components,
            )
            if float(b.mean()) < min_fill:
                b = _boost_center_disk(b, n=n)
                b = _process_paper_like_binary(b, cfg, min_area=min_area, max_components=max_components)

            rho, lifted = _paper_like_lifted_view(b)
            if _paper_like_is_valid_lifted(lifted, min_fill=min_fill, max_fill=max_fill):
                break

        if not _paper_like_is_valid_lifted(lifted, min_fill=min_fill, max_fill=max_fill):
            rho, lifted, symmetry, primitives_used, fallback_steps = _paper_like_force_nonempty(
                rng,
                cfg,
                min_fill=min_fill,
                max_fill=max_fill,
                symmetry_candidates=symmetry_candidates,
                min_area=min_area,
                max_components=max_components,
            )
            steps.extend(fallback_steps)
        else:
            steps.extend(
                [
                    *([f"symmetry:{symmetry}"] if symmetry != "none" else []),
                    "morphology_opening" if bool(cfg.get("morphology_opening", True)) else "",
                    "morphology_closing" if bool(cfg.get("morphology_closing", True)) else "",
                    "fill_holes" if bool(cfg.get("fill_holes", True)) else "",
                    f"compact_cleanup_min_area={min_area}",
                    f"primitive_draws={len(primitives_used)}",
                ]
            )
            if max_components is not None and max_components > 0:
                steps.append(f"limit_components_max={max_components}")
        steps = [step for step in steps if step]

        stats = _binary_stats(lifted)
        return StructureSample(
            rho=rho.astype(np.float32),
            binary_grid=lifted,
            metadata={
                "symmetry": symmetry,
                "generator_name": self.name,
                "num_primitives": len(primitives_used),
                "primitive_types": primitives_used,
                "num_connected_components": int(stats["num_connected_components"]),
            },
            postprocess_steps=steps,
        )


class PaperLikeParametricGenerator(_GeneratorBase):
    """Parametric, paper-style photonic crystal unit cells (strict C4 templates)."""

    _TEMPLATE_NAMES = (
        "center_block",
        "corner_pads_plus_center",
        "ring_frame",
        "cross_void",
        "frame_with_notches",
        "center_plus_satellites",
    )

    def __init__(self, *, options: Mapping[str, Any] | None = None) -> None:
        super().__init__(name="paper_like_parametric")
        self.options = dict(options or {})

    def _cfg(self, dataset_cfg: Mapping[str, Any]) -> dict[str, Any]:
        cfg = dict(dataset_cfg)
        cfg.update(self.options)
        cfg.setdefault(
            "template_weights",
            {
                "center_block": 0.08,
                "corner_pads_plus_center": 0.20,
                "ring_frame": 0.20,
                "cross_void": 0.18,
                "frame_with_notches": 0.16,
                "center_plus_satellites": 0.18,
            },
        )
        cfg.setdefault("max_variation_ops", 2)
        cfg.setdefault("min_component_area", 4)
        cfg.setdefault("max_hole_area", 4)
        return cfg

    def _template_center_block(self, rng: np.random.Generator, cfg: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        g = np.zeros((16, 16), dtype=np.float32)
        core_width = int(rng.choice(np.array([4, 6, 8], dtype=np.int64)))
        _centered_square(g, core_width, value=1.0)
        hole_width = 0
        if bool(cfg.get("allow_center_hole", True)) and float(rng.random()) < 0.45:
            hole_width = int(rng.choice(np.array([2, 4], dtype=np.int64)))
            if hole_width < core_width - 2:
                _centered_square(g, hole_width, value=0.0)
        return g, {
            "template_family": "center_block",
            "template_params": {"core_width": core_width, "hole_width": hole_width},
            "bridge_width": 0,
            "frame_thickness": 0,
            "component_count_target": 1,
        }

    def _template_corner_pads_plus_center(
        self, rng: np.random.Generator, cfg: Mapping[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        g = np.zeros((16, 16), dtype=np.float32)
        center_width = int(rng.choice(np.array([4, 6], dtype=np.int64)))
        pad_size = int(rng.choice(np.array([2, 4], dtype=np.int64)))
        inset = int(rng.choice(np.array([1, 2, 3], dtype=np.int64)))
        bridge_width = int(rng.choice(np.array([0, 2, 4], dtype=np.int64))) if bool(cfg.get("allow_bridges", True)) else 0
        _centered_square(g, center_width, value=1.0)
        if bool(cfg.get("allow_corner_pads", True)):
            _corner_pads(g, pad_size, inset, value=1.0)
        if bridge_width >= 2:
            _cardinal_bridges(g, center_width, 16 - 2 * inset, bridge_width, value=1.0)
        return g, {
            "template_family": "corner_pads_plus_center",
            "template_params": {
                "center_width": center_width,
                "pad_size": pad_size,
                "inset": inset,
                "bridge_width": bridge_width,
            },
            "bridge_width": bridge_width,
            "frame_thickness": 0,
            "component_count_target": 5 if bridge_width == 0 else 1,
        }

    def _template_ring_frame(self, rng: np.random.Generator, cfg: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        g = np.zeros((16, 16), dtype=np.float32)
        outer_width = int(rng.choice(np.array([8, 10, 12], dtype=np.int64)))
        thickness = int(rng.choice(np.array([2, 4], dtype=np.int64)))
        _, _, frame_thickness = _ring_frame(g, outer_width, thickness, value=1.0)
        center_width = int(rng.choice(np.array([2, 4], dtype=np.int64)))
        if float(rng.random()) < 0.75:
            _centered_square(g, center_width, value=1.0)
        return g, {
            "template_family": "ring_frame",
            "template_params": {
                "outer_width": outer_width,
                "frame_thickness": frame_thickness,
                "center_width": center_width,
            },
            "bridge_width": 0,
            "frame_thickness": frame_thickness,
            "component_count_target": 2,
        }

    def _template_cross_void(self, rng: np.random.Generator, cfg: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        g = np.zeros((16, 16), dtype=np.float32)
        outer_width = int(rng.choice(np.array([8, 10, 12], dtype=np.int64)))
        cross_width = int(rng.choice(np.array([2, 4], dtype=np.int64)))
        _centered_square(g, outer_width, value=1.0)
        _center_cross_void(g, cross_width, outer_width - 2)
        if bool(cfg.get("allow_corner_pads", True)) and float(rng.random()) < 0.65:
            _corner_pads(g, 2, 2, value=1.0)
        return g, {
            "template_family": "cross_void",
            "template_params": {
                "outer_width": outer_width,
                "cross_width": cross_width,
            },
            "bridge_width": 0,
            "frame_thickness": 0,
            "component_count_target": 1,
        }

    def _template_frame_with_notches(
        self, rng: np.random.Generator, cfg: Mapping[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        g = np.zeros((16, 16), dtype=np.float32)
        outer_width = int(rng.choice(np.array([10, 12, 14], dtype=np.int64)))
        thickness = 2
        outer_s, outer_e, frame_thickness = _ring_frame(g, outer_width, thickness, value=1.0)
        notch_width = int(rng.choice(np.array([2, 4], dtype=np.int64)))
        notch_depth = int(rng.choice(np.array([1, 2], dtype=np.int64)))
        _side_notches(g, outer_s, outer_e, frame_thickness, notch_width, notch_depth)
        if float(rng.random()) < 0.75:
            _centered_square(g, 2, value=1.0)
        return g, {
            "template_family": "frame_with_notches",
            "template_params": {
                "outer_width": outer_width,
                "frame_thickness": frame_thickness,
                "notch_width": notch_width,
                "notch_depth": notch_depth,
            },
            "bridge_width": 0,
            "frame_thickness": frame_thickness,
            "component_count_target": 2,
        }

    def _template_center_plus_satellites(
        self, rng: np.random.Generator, cfg: Mapping[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        g = np.zeros((16, 16), dtype=np.float32)
        center_width = int(rng.choice(np.array([4, 6], dtype=np.int64)))
        satellite_size = int(rng.choice(np.array([2, 4], dtype=np.int64)))
        offset = int(rng.choice(np.array([1, 2, 3], dtype=np.int64)))
        _centered_square(g, center_width, value=1.0)
        _diagonal_satellites(g, satellite_size, offset, value=1.0)
        bridge_width = 0
        if bool(cfg.get("allow_bridges", True)) and float(rng.random()) < 0.45:
            bridge_width = 2
            _cardinal_bridges(g, center_width, center_width + 2 * offset + 2 * satellite_size, bridge_width, value=1.0)
        return g, {
            "template_family": "center_plus_satellites",
            "template_params": {
                "center_width": center_width,
                "satellite_size": satellite_size,
                "offset": offset,
                "bridge_width": bridge_width,
            },
            "bridge_width": bridge_width,
            "frame_thickness": 0,
            "component_count_target": 5 if bridge_width == 0 else 1,
        }

    def _generate_candidate(self, rng: np.random.Generator, cfg: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        weight_map = cfg.get("template_weights", {})
        probs = _normalized_weights(weight_map, self._TEMPLATE_NAMES)
        name = str(rng.choice(np.array(self._TEMPLATE_NAMES, dtype=object), p=probs))
        if name == "center_block":
            grid, meta = self._template_center_block(rng, cfg)
        elif name == "corner_pads_plus_center":
            grid, meta = self._template_corner_pads_plus_center(rng, cfg)
        elif name == "ring_frame":
            grid, meta = self._template_ring_frame(rng, cfg)
        elif name == "cross_void":
            grid, meta = self._template_cross_void(rng, cfg)
        elif name == "frame_with_notches":
            grid, meta = self._template_frame_with_notches(rng, cfg)
        else:
            grid, meta = self._template_center_plus_satellites(rng, cfg)
        varied, ops = _apply_paper_like_variations(grid, rng, max_ops=int(cfg.get("max_variation_ops", 2)))
        meta = {**meta, "variation_ops": ops}
        return varied, meta

    def sample(self, *, rng: np.random.Generator, dataset_cfg: Mapping[str, Any]) -> StructureSample:
        cfg = self._cfg(dataset_cfg)
        min_fill, max_fill = _resolve_fill_fraction(cfg, default_low=0.18, default_high=0.62)
        max_components = int(cfg.get("max_components", 5))
        min_low_freq_ratio = float(cfg.get("min_low_frequency_energy_ratio", 0.55))

        last_grid = np.zeros((16, 16), dtype=np.float32)
        last_meta: dict[str, Any] = {}
        for _ in range(96):
            grid, meta = self._generate_candidate(rng, cfg)
            binary = np.clip((grid >= 0.5).astype(np.float32), 0.0, 1.0)
            binary = _cleanup_paper_like_binary(
                binary,
                min_component_area=int(cfg.get("min_component_area", 4)),
                max_hole_area=int(cfg.get("max_hole_area", 4)),
            )
            lifted = lift_rho_to_grid(_wedge_rho_from_grid(binary)).astype(np.float32)
            if not np.array_equal(binary, lifted):
                continue
            if not _paper_like_stats_ok(
                binary,
                min_fill=min_fill,
                max_fill=max_fill,
                max_components=max_components,
                min_low_freq_ratio=min_low_freq_ratio,
            ):
                last_grid = binary
                last_meta = meta
                continue
            return StructureSample(
                rho=_wedge_rho_from_grid(binary),
                binary_grid=binary,
                metadata={
                    "symmetry": "c4",
                    "generator_name": self.name,
                    **meta,
                },
                postprocess_steps=[
                    "paper_like_parametric",
                    f"template:{meta['template_family']}",
                    "symmetry:c4",
                    *[f"variation:{name}" for name in meta.get("variation_ops", [])],
                    "cleanup:small_components",
                    "cleanup:small_holes",
                    "reject_filter:paper_like",
                ],
            )

        fallback = lift_rho_to_grid(_wedge_rho_from_grid(last_grid)).astype(np.float32)
        binary = fallback if float(np.sum(fallback)) > 0.0 else np.zeros((16, 16), dtype=np.float32)
        if float(np.sum(binary)) <= 0.0:
            _centered_square(binary, 4, value=1.0)
            last_meta = {
                "template_family": "center_block",
                "template_params": {"core_width": 4, "hole_width": 0},
                "bridge_width": 0,
                "frame_thickness": 0,
                "component_count_target": 1,
            }
        return StructureSample(
            rho=_wedge_rho_from_grid(binary),
            binary_grid=binary,
            metadata={
                "symmetry": "c4",
                "generator_name": self.name,
                **last_meta,
            },
            postprocess_steps=[
                "paper_like_parametric",
                f"template:{last_meta.get('template_family', 'fallback')}",
                "symmetry:c4",
                "fallback_after_rejects",
            ],
        )


class PhotonicCrystalUnitCellGenerator(_GeneratorBase):
    """Parametric photonic crystal style unit-cell templates."""

    def __init__(self) -> None:
        super().__init__(name="photonic_crystal_unit_cell")

    def sample(self, *, rng: np.random.Generator, dataset_cfg: Mapping[str, Any]) -> StructureSample:
        n = 16
        g = np.ones((n, n), dtype=np.float32)
        steps: list[str] = ["photonic_crystal_unit_cell"]

        symmetry = str(_randomize_symmetry(rng, ("c4", "c2", "mirror", "none")))
        min_fill, max_fill = _resolve_fill_fraction(dataset_cfg, default_low=0.10, default_high=0.80)

        n_holes = int(rng.integers(1, 5))
        for _ in range(max(1, n_holes)):
            hole_type = rng.choice(["circle", "ellipse", "rectangle", "ring", "cross"])
            if hole_type == "ellipse":
                yy, xx = np.ogrid[:n, :n]
                cx = float(rng.uniform(0.2, 0.8) * (n - 1))
                cy = float(rng.uniform(0.2, 0.8) * (n - 1))
                rx = float(rng.uniform(n / 12.0, n / 5.0))
                ry = float(rng.uniform(n / 14.0, n / 6.0))
                rr = ((yy - cy) / max(ry, 1e-12)) ** 2 + ((xx - cx) / max(rx, 1e-12)) ** 2
                g[rr <= 1.0] = 0.0
                steps.append("ellipse_hole")
            else:
                _draw_primitive_raster(g, "rectangle" if hole_type == "rectangle" else hole_type, rng, invert=True)
                steps.append(f"{hole_type}_hole")

            if g.mean() < min_fill:
                break

        if symmetry != "none":
            g = 1.0 - _apply_symmetry_projection(1.0 - g, symmetry=symmetry)
            steps.append(f"symmetry:{symmetry}")
            g = np.clip(g, 0.0, 1.0)

        if g.mean() < min_fill:
            steps.append("fallback_fill_center")
            y0 = int(n * 0.3)
            g[y0 : n - y0, y0 : n - y0] = 1.0
        if g.mean() > max_fill:
            factor = max_fill / max(float(g.mean()), 1e-12)
            g = np.clip(g * factor, 0.0, 1.0)
            steps.append(f"occupancy_scale={factor:.4f}")

        b = (g >= 0.5).astype(np.float32)
        rho = _wedge_rho_from_grid(b)
        return StructureSample(
            rho=rho,
            binary_grid=b,
            metadata={
                "symmetry": symmetry,
            },
            postprocess_steps=steps,
        )


class MorphologyFilteredGenerator(_GeneratorBase):
    """Apply morphology/cleanup after a base generator."""

    def __init__(self, *, base_generator: _GeneratorBase) -> None:
        super().__init__(name="morphology_filtered")
        self.base_generator = base_generator

    def sample(self, *, rng: np.random.Generator, dataset_cfg: Mapping[str, Any]) -> StructureSample:
        base = self.base_generator.sample(rng=rng, dataset_cfg=dataset_cfg)
        grid = base.binary_grid

        from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, generate_binary_structure

        structure = generate_binary_structure(2, 1)
        g = binary_opening(grid > 0.5, structure=structure)
        g = binary_closing(g, structure=structure)
        g = binary_fill_holes(g)
        g = g.astype(np.float32)

        return StructureSample(
            rho=_wedge_rho_from_grid(g),
            binary_grid=g,
            metadata={**base.metadata},
            postprocess_steps=base.postprocess_steps + ["morphology_opening", "morphology_closing", "morphology_fill_holes"],
        )


class MixedDatasetGenerator(_GeneratorBase):
    """Weighted mixture of sub-generators."""

    def __init__(self, *, children: list[tuple[_GeneratorBase, float]]) -> None:
        super().__init__(name="mixed_dataset")
        if not children:
            raise ValueError("mixed_dataset requires at least one child generator")
        total = float(sum(max(0.0, w) for _, w in children))
        if total <= 0:
            raise ValueError("mixed_dataset child weights must sum to > 0")
        self.children = children
        self._weights = np.cumsum([max(0.0, w) / total for _, w in children])

    def sample(self, *, rng: np.random.Generator, dataset_cfg: Mapping[str, Any]) -> StructureSample:
        p = float(rng.random())
        idx = int(np.searchsorted(self._weights, p, side="right"))
        idx = min(idx, len(self.children) - 1)
        child, _ = self.children[idx]
        child_sample = child.sample(rng=rng, dataset_cfg=dataset_cfg)
        return StructureSample(
            rho=child_sample.rho,
            binary_grid=child_sample.binary_grid,
            metadata={**child_sample.metadata, "selected_child": child.name},
            postprocess_steps=child_sample.postprocess_steps + [f"mixed_from:{child.name}"],
        )


def build_structure_generator(spec_in: Mapping[str, Any] | str) -> _GeneratorBase:
    """Build a structure generator from a name, spec mapping, or dataset config."""

    if isinstance(spec_in, str):
        name = spec_in
        params: dict[str, Any] = {}
    elif isinstance(spec_in, Mapping):
        if "structure_generator" in spec_in:
            spec = spec_in["structure_generator"]
        elif "name" in spec_in:
            spec = spec_in
        else:
            spec = "legacy_random"

        if isinstance(spec, str):
            name = spec
            params = {}
        elif isinstance(spec, Mapping):
            name = str(spec.get("name", "legacy_random"))
            params = dict(spec)
        else:
            name = "legacy_random"
            params = {}
    else:
        name = "legacy_random"
        params = {}

    if name == "legacy_random":
        return LegacyRandomGenerator()
    if name == "paper_like_compact":
        return PaperLikeCompactGenerator(options=params)
    if name == "paper_like_parametric":
        return PaperLikeParametricGenerator(options=params)
    if name == "photonic_crystal_unit_cell":
        return PhotonicCrystalUnitCellGenerator()
    if name == "morphology_filtered":
        base = params.get("base", "legacy_random")
        base_generator = build_structure_generator({"structure_generator": base})
        return MorphologyFilteredGenerator(base_generator=base_generator)
    if name == "mixed_dataset":
        components = params.get("components", params.get("weights", {}))
        if not isinstance(components, Mapping) or not components:
            raise ValueError("mixed_dataset requires a mapping in components/weights")

        children: list[tuple[_GeneratorBase, float]] = []
        for child_name, weight in components.items():
            child_spec = {"name": str(child_name)}
            child_gen = build_structure_generator(child_spec)
            children.append((child_gen, float(weight)))
        return MixedDatasetGenerator(children=children)

    raise ValueError(f"Unknown structure_generator: {name}")


def collect_structure_metadata(
    *,
    binary_grid: np.ndarray,
    seed: int,
    sample_size: int,
    generator_name: str,
    symmetry: str,
    postprocess_steps: list[str],
) -> dict[str, Any]:
    """Build unified per-sample metadata for logging and statistics."""

    stat = _binary_stats(binary_grid)
    out: dict[str, Any] = {
        "seed": int(seed),
        "grid_size": int(sample_size),
        "generator_name": generator_name,
        "symmetry": symmetry,
        "postprocess_steps": list(postprocess_steps),
        "symmetry_error": _symmetry_error(binary_grid, symmetry),
    }
    out.update(stat)
    return out


__all__ = [
    "StructureSample",
    "_GeneratorBase",
    "build_structure_generator",
    "collect_structure_metadata",
    "LegacyRandomGenerator",
    "PaperLikeCompactGenerator",
    "PaperLikeParametricGenerator",
    "PhotonicCrystalUnitCellGenerator",
    "MorphologyFilteredGenerator",
    "MixedDatasetGenerator",
]
