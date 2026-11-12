"""Approved planar PCSEL motifs and deterministic geometry-only test plans.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
import math
from typing import Any, Mapping

import numpy as np

PCSEL_FAMILIES = ("S01", "S02", "D01", "D02", "D03", "D04", "D05", "T01", "T02", "T03", "Q01", "U01")
PCSEL_ALGORITHM = "phc2d.analytic_pcsel.v1"


def pcsel_parameter_rows(family: str) -> list[dict[str, float]]:
    """Return exactly the 2000 parameter points reviewed in the geometry atlas."""
    def linear(a: float, b: float, n: int) -> list[float]:
        return [float(v) for v in np.linspace(a, b, n)]
    if family == "S01":
        return [dict(f=f, eta=e, theta=t) for f, e, t in product(linear(.10, .20, 100), [1.2, 1.5, 1.8, 2., 2.4], [0., 15., 30., 45.])]
    if family == "S02":
        return [dict(f=f, theta=t) for f, t in product(linear(.10, .20, 200), linear(0, 45, 10))]
    if family == "D01":
        return [dict(f=f, w=w, d=d, c=.8) for f, w, d in product(linear(.06, .12, 100), [.53, 5/9, .58, .61, .64], [.235, .25, .257, .28])]
    if family == "D02":
        return [dict(f=f, w=.5, d=d, c=c) for f, d, c in product(linear(.06, .12, 100), [.22, .235, .25, .257, .28], [.65, .7, .75, .8])]
    if family == "D03":
        return [dict(f=f, w=9/16, d=d, c=c) for f, d, c in product(linear(.06, .12, 100), [.235, .25, .257, .27, .28], [.4, .45, .5, .55])]
    if family == "D04":
        return [dict(f=f, w=w, c=c) for f, c, w in product(linear(.10, .20, 100), [.4, .5, .6, .7, .8], [.45, .50, .55, .60])]
    if family == "D05":
        return [dict(f=f, w=w, d=d) for f, w, d in product(linear(.10, .20, 100), [.30, .35, .40, .45, .50], [.42, .45, .48, .50])]
    if family == "T01":
        return [dict(f=f) for f in linear(.08, .20, 2000)]
    if family == "T02":
        return [dict(f=f, delta_x=-u, delta_y=-u*v) for f, u, v in product(linear(.08, .20, 100), [.02, .03, .04, .05, .06], [.25, .5, .75, 1.])]
    if family == "T03":
        return [dict(f=f, t=t, s=s) for f, t, s in product(linear(.06, .12, 100), [-.01, -.005, 0, .005, .01], [.8, .9, 1., 1.1])]
    if family == "Q01":
        return [dict(f=f, s=s, delta=d) for f, s, d in product(linear(.10, .20, 100), [.55, .65, .75, .85, .95], [0., .01, .02, .03])]
    if family == "U01":
        return [dict(f=f, kappa=k, eta=e, theta=-30.) for f, k, e in product(linear(.08, .20, 100), [1., 1.1, 1.2, 1.3, 1.4], [.5, .65, .8, .95])]
    raise ValueError(f"Unknown PCSEL family: {family}")


def _ellipse(area: float, center: tuple[float, float], ratio: float = 1., angle: float = 0.) -> dict[str, Any]:
    a = math.sqrt(area / (math.pi * ratio))
    radians = math.radians(angle)
    return dict(kind="ellipse", center=list(center), a=a, b=a*ratio,
                cos=math.cos(radians), sin=math.sin(radians))


@lru_cache(maxsize=32)
def _three_disc_union(kappa: float, eta: float) -> tuple[float, float]:
    """Exact circular-arc area and quadrature centroid of the unit-radius union."""
    h = eta*kappa
    centers = [(-kappa/2, -h/3), (kappa/2, -h/3), (0., 2*h/3)]
    nodes, weights = np.polynomial.legendre.leggauss(32)
    area = moment_y = 0.
    for i, (cx, cy) in enumerate(centers):
        angles = [0., 2*math.pi]
        for j, (dx, dy) in enumerate(centers):
            if i == j:
                continue
            distance = math.hypot(dx-cx, dy-cy)
            if not 0 < distance < 2:
                raise ValueError("U01 requires intersecting distinct unit circles")
            mid = math.atan2(dy-cy, dx-cx)
            half = math.acos(distance/2)
            angles.extend([(mid-half) % (2*math.pi), (mid+half) % (2*math.pi)])
        angles.sort()
        for lo, hi in zip(angles, angles[1:]):
            mid = (lo+hi)/2
            px, py = cx+math.cos(mid), cy+math.sin(mid)
            if any((px-dx)**2+(py-dy)**2 < 1-1e-12 for j, (dx, dy) in enumerate(centers) if j != i):
                continue
            area += .5*((hi-lo)+cx*(math.sin(hi)-math.sin(lo))+cy*(math.cos(lo)-math.cos(hi)))
            t = (lo+hi)/2+(hi-lo)/2*nodes
            # Integral y dA = -1/2 integral y^2 dx on the exposed CCW arcs.
            moment_y += .5*(hi-lo)/2*float(np.dot(weights, (cy+np.sin(t))**2*np.sin(t)))
    return float(area), float(moment_y/area)


def pcsel_primitives(family: str, p: Mapping[str, float]) -> list[dict[str, Any]]:
    """Construct exact ellipses or polygon vertices from approved raw parameters."""
    f = float(p["f"])
    if family in {"S01", "S02"}:
        if family == "S01":
            h, b = math.sqrt(2*f*p["eta"]), math.sqrt(2*f/p["eta"])
            vertices = [(2*h/3, 0), (-h/3, b/2), (-h/3, -b/2)]
        else:
            length = math.sqrt(2*f)
            vertices = [(-length/3, -length/3), (2*length/3, -length/3), (-length/3, 2*length/3)]
        angle = math.radians(p["theta"])
        cs, sn = math.cos(angle), math.sin(angle)
        return [dict(kind="polygon", vertices=[[cs*x-sn*y, sn*x+cs*y] for x, y in vertices])]
    if family in {"D01", "D02", "D03"}:
        d, w, c = p["d"], p["w"], p["c"]
        return [_ellipse(f*w, (-d/2, d/2), c, 45), _ellipse(f*(1-w), (d/2, -d/2), 1. if family == "D03" else c, 45)]
    if family == "D04":
        return [_ellipse(f*p["w"], (-.25, 0), p["c"], 90), _ellipse(f*(1-p["w"]), (.25, 0))]
    if family == "D05":
        d = p["d"]
        return [_ellipse(f*p["w"], (-d/2, -d/2)), _ellipse(f*(1-p["w"]), (d/2, d/2))]
    if family in {"T01", "T02"}:
        centers = [(-.25, -.25), (.25, -.25), (.25+p.get("delta_x", 0), .25+p.get("delta_y", 0))]
        return [_ellipse(f/3, center) for center in centers]
    if family == "T03":
        centers = [(-.37+p["t"], -.37+p["t"]), (0, 0), (.3, .3)]
        return [_ellipse(f*w/(2+p["s"]), center) for w, center in zip([1, p["s"], 1], centers)]
    if family == "Q01":
        centers = [(-.25, -.25), (.25, -.25), (.25, .25), (-.25+p["delta"], .25)]
        return [_ellipse(f*w/(2.4+p["s"]), center) for w, center in zip([1, .8, p["s"], .6], centers)]
    if family == "U01":
        d, h = p["kappa"], p["eta"]*p["kappa"]
        area, cy = _three_disc_union(d, p["eta"])
        scale = math.sqrt(f/area)
        angle = math.radians(p["theta"])
        cs, sn = math.cos(angle), math.sin(angle)
        centers = [(-d/2, -h/3-cy), (d/2, -h/3-cy), (0, 2*h/3-cy)]
        return [_ellipse(math.pi*scale**2, (scale*(cs*x-sn*y), scale*(sn*x+cs*y))) for x, y in centers]
    raise ValueError(f"Unknown PCSEL family: {family}")


def build_pcsel_test_plan() -> list[dict[str, Any]]:
    """Build the approved twelve-family 24000-case planar geometry plan."""
    return [dict(sample_id=f"pcsel_{family}_{i:04d}", parameters=dict(
                shape=family, air_fraction=p["f"], raw=p, primitives=pcsel_primitives(family, p)))
            for family in PCSEL_FAMILIES for i, p in enumerate(pcsel_parameter_rows(family))]


def evaluate_pcsel(x: Any, y: Any, parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate periodic binary material, with analytic scalar MPB callbacks."""
    xx, yy = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    xx, yy = (xx+.5) % 1-.5, (yy+.5) % 1-.5
    scalar = xx.ndim == 0
    air = False if scalar else np.zeros_like(xx, dtype=bool)
    for primitive in parameters["primitives"]:
        if primitive["kind"] == "ellipse":
            cx, cy = primitive["center"]
            dx, dy = (xx-cx+.5) % 1-.5, (yy-cy+.5) % 1-.5
            cs, sn, a, b = (primitive[k] for k in ("cos", "sin", "a", "b"))
            inside = ((cs*dx+sn*dy)/a)**2+((-sn*dx+cs*dy)/b)**2 <= 1
            air |= inside
        else:
            v = primitive["vertices"]
            for tx, ty in product([-1, 0, 1], repeat=2):
                dx, dy = xx-tx, yy-ty
                inside = True if scalar else np.ones_like(xx, dtype=bool)
                for i, (vx, vy) in enumerate(v):
                    wx, wy = v[(i+1) % len(v)]
                    inside &= (wx-vx)*(dy-vy)-(wy-vy)*(dx-vx) >= -1e-14
                air |= inside
        if scalar and bool(air):
            return np.asarray(0.)
    return np.asarray(1.-air)


def _support(primitive: Mapping[str, Any], direction: np.ndarray) -> float:
    if primitive["kind"] == "polygon":
        return float(np.max(np.asarray(primitive["vertices"]) @ direction))
    cs, sn, a, b = (primitive[k] for k in ("cos", "sin", "a", "b"))
    nx, ny = direction
    return float(np.dot(primitive["center"], direction)+math.hypot(a*(cs*nx+sn*ny), b*(-sn*nx+cs*ny)))


def certify_pcsel_clearance(parameters: Mapping[str, Any]) -> float:
    """Return a conservative separating-axis gap certificate including periodic images."""
    gs = parameters["primitives"]
    fused = parameters["shape"] == "U01"
    gap = math.inf
    for i, a in enumerate(gs):
        ca = np.asarray(a.get("center", [0, 0]), dtype=float)
        for j, b in enumerate(gs):
            cb = np.asarray(b.get("center", [0, 0]), dtype=float)
            for tx, ty in product([-1, 0, 1], repeat=2):
                if tx == ty == 0 and (i == j or fused):
                    continue
                shift = np.asarray([tx, ty], dtype=float)
                vector = cb+shift-ca
                direction = vector/np.linalg.norm(vector)
                directions = [direction]
                for primitive in [a, b]:
                    if primitive["kind"] == "polygon":
                        vertices = np.asarray(primitive["vertices"])
                        for edge in np.roll(vertices, -1, axis=0)-vertices:
                            normal = np.array([-edge[1], edge[0]])/np.linalg.norm(edge)
                            directions.extend([normal, -normal])
                bound = max(float(np.dot(shift, axis)-_support(a, axis)-_support(b, -axis)) for axis in directions)
                gap = min(gap, bound)
    return gap


def validate_pcsel_config(cfg: Mapping[str, Any]) -> list[str]:
    """Fail closed on a changed sample, geometry, or authorization contract."""
    errors = []
    generator, dataset = cfg["generator"], cfg["dataset"]
    if list(generator.get("families", [])) != list(PCSEL_FAMILIES):
        errors.append("PCSEL families must match the approved twelve-family atlas")
    if generator.get("samples_per_family") != 2000 or dataset.get("sample_count") != 24000:
        errors.append("PCSEL contract requires twelve families of 2000 cases")
    if dict(dataset.get("split", {})) != dict(train=0, val=0, test=24000):
        errors.append("PCSEL dataset must be test-only")
    if not cfg.get("authorization", {}).get("formal_generation_authorized", False):
        errors.append("PCSEL formal generation needs explicit authorization")
    if generator.get("algorithm_version") != PCSEL_ALGORITHM:
        errors.append("PCSEL algorithm version mismatch")
    return errors
