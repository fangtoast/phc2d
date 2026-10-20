"""Area-controlled analytic single-hole geometries for frozen transfer tests.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from phc2d.data.full_cell_fourier import wrapped_periodic_delta

SINGLE_HOLE_FAMILIES = ("circle", "ellipse", "triangle", "square", "rectangle", "hexagon")


def single_hole_dimensions(parameters: Mapping[str, Any]) -> tuple[float, float]:
    """Return half axes or polygon circumradius for a unit-area periodic cell."""
    shape = str(parameters["shape"])
    fraction = float(parameters["air_fraction"])
    ratio = float(parameters.get("aspect_ratio", 1.0))
    if shape not in SINGLE_HOLE_FAMILIES:
        raise ValueError(f"unsupported single-hole shape: {shape}")
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("air_fraction must be finite and in (0,1)")
    if not math.isfinite(ratio) or ratio < 1:
        raise ValueError("aspect_ratio must be finite and at least one")
    if shape not in {"ellipse", "rectangle"} and ratio != 1:
        raise ValueError("only ellipse and rectangle accept an aspect ratio")
    if shape in {"circle", "ellipse"}:
        first, second = math.sqrt(fraction * ratio / math.pi), math.sqrt(fraction / (ratio * math.pi))
        enclosing_radius = first
    elif shape in {"square", "rectangle"}:
        first, second = math.sqrt(fraction * ratio) / 2, math.sqrt(fraction / ratio) / 2
        enclosing_radius = math.hypot(first, second)
    else:
        sides = 3 if shape == "triangle" else 6
        first = second = math.sqrt(2 * fraction / (sides * math.sin(2 * math.pi / sides)))
        enclosing_radius = first
    if enclosing_radius >= .49:
        raise ValueError("hole requires a positive periodic inter-hole clearance")
    return first, second


def evaluate_single_hole(x: Any, y: Any, parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate dielectric material weight; one connected air hole per cell."""
    first, second = single_hole_dimensions(parameters)
    angle = float(parameters.get("angle_deg", 0.0))
    center = tuple(float(v) for v in parameters.get("center", [0.0, 0.0]))
    if len(center) != 2 or not np.isfinite([angle, *center]).all():
        raise ValueError("finite angle and two-coordinate center required")
    dx, dy = np.broadcast_arrays(wrapped_periodic_delta(np.asarray(x), center[0]),
                                 wrapped_periodic_delta(np.asarray(y), center[1]))
    radians = math.radians(angle)
    xr = math.cos(radians) * dx + math.sin(radians) * dy
    yr = -math.sin(radians) * dx + math.cos(radians) * dy
    shape = parameters["shape"]
    if shape in {"circle", "ellipse"}:
        air = (xr / first) ** 2 + (yr / second) ** 2 <= 1
    elif shape in {"square", "rectangle"}:
        air = (np.abs(xr) <= first) & (np.abs(yr) <= second)
    else:
        sides = 3 if shape == "triangle" else 6
        vertices = [(first * math.cos(2 * math.pi * i / sides), first * math.sin(2 * math.pi * i / sides)) for i in range(sides)]
        air = np.ones(xr.shape, dtype=bool)
        for i, (vx, vy) in enumerate(vertices):
            wx, wy = vertices[(i + 1) % sides]
            air &= (wx - vx) * (yr - vy) - (wy - vy) * (xr - vx) >= -1e-14
    return (~air).astype(np.float64)


def build_single_hole_test_plan(*, samples_per_family: int = 200) -> list[dict[str, Any]]:
    """Return a deterministic balanced plan, densifying only air-fraction levels."""
    samples = int(samples_per_family)
    if samples <= 0 or samples % 20 != 0:
        raise ValueError("samples_per_family must be a positive multiple of 20")
    rows: list[dict[str, Any]] = []
    # Angle ranges quotient shape rotations and square-lattice D4 symmetry.
    # They define a parameter scan, not an iid population of manufactured devices.
    for shape in SINGLE_HOLE_FAMILIES:
        if shape == "circle":
            combinations = [(f, 1.0, 0.0) for f in np.linspace(.10, .20, samples)]
        elif shape in {"ellipse", "rectangle"}:
            combinations = [(f, q, angle) for f in np.linspace(.10, .20, samples // 20)
                            for q in [1.25, 1.5, 2.0, 3.0] for angle in np.linspace(0, 45, 5)]
        else:
            maximum_angle = 15 if shape in {"triangle", "hexagon"} else 45
            combinations = [(f, 1.0, angle) for f in np.linspace(.10, .20, samples // 10)
                            for angle in np.linspace(0, maximum_angle, 10)]
        for index, (fraction, ratio, angle) in enumerate(combinations):
            parameters = {"shape": shape, "air_fraction": float(fraction), "aspect_ratio": float(ratio),
                          "angle_deg": float(angle), "center": [0.0, 0.0]}
            single_hole_dimensions(parameters)
            rows.append({"sample_id": f"single_{shape}_{index:04d}", "parameters": parameters})
    return rows
