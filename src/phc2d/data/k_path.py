"""Versioned physical k-path metadata and sidecar helpers.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SCHEMA_VERSION = "phc2d.k_path.v1"
SIDECAR_SUFFIX = ".kpath.v1.json"
MANIFEST_SUFFIX = ".manifest.yaml"
VERIFIED_K_PATH_STATUSES = frozenset(
    {
        "physically_verified",
        "physically_verified_with_documented_numerical_tolerance",
    }
)


@dataclass(frozen=True)
class KPathNode:
    """A named high-symmetry or path-control node."""

    label: str
    coordinate: tuple[float, ...]


@dataclass(frozen=True)
class KPathSegment:
    """A segment in a sampled k path."""

    start_node: str
    end_node: str
    start_index: int
    end_index: int
    number_of_points: int
    endpoint_policy: str


@dataclass(frozen=True)
class KPathMetadata:
    """Source and verification boundary for a k-path sidecar."""

    source_type: str = ""
    source_path: str = ""
    source_commit: str = ""
    source_run_id: str = ""
    source_config_hash: str = ""
    verification_status: str = "unverified"
    verification_run_id: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class KPathSpec:
    """Complete sampled k-path contract for query-form datasets."""

    schema_version: str
    k_path_id: str
    coordinate_system: str
    coordinate_dimension: int
    reciprocal_basis_convention: str
    nodes: tuple[KPathNode, ...]
    segments: tuple[KPathSegment, ...]
    k_points: tuple[tuple[float, ...], ...]
    k_index: tuple[int, ...]
    path_s: tuple[float, ...]
    segment_id: tuple[int, ...]
    segment_local_t: tuple[float, ...]
    node_indices: dict[str, int]
    duplicate_coordinate_groups: tuple[tuple[int, ...], ...]
    Nk: int
    metadata: KPathMetadata = field(default_factory=KPathMetadata)
    k_points_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["nodes"] = [asdict(node) for node in self.nodes]
        payload["segments"] = [asdict(segment) for segment in self.segments]
        payload["metadata"] = asdict(self.metadata)
        return payload


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256(payload: Any) -> str:
    data = payload if isinstance(payload, bytes) else _canonical_json(payload)
    return hashlib.sha256(data).hexdigest()


def _as_tuple_points(points: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("k_points must be a 2D array-like object")
    return tuple(tuple(float(v) for v in row) for row in arr)


def hash_k_path_points(points: Sequence[Sequence[float]], *, decimals: int = 12) -> str:
    """Hash sampled coordinates after stable rounding."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("k_points must be 2D")
    rounded = np.round(arr, int(decimals)).tolist()
    return _sha256(rounded)


def hash_k_path_spec(spec: KPathSpec) -> str:
    """Hash a KPathSpec excluding self-referential hashes and verification state."""
    payload = spec.to_dict()
    payload.pop("k_points_hash", None)
    metadata = dict(payload.get("metadata") or {})
    metadata.pop("verification_status", None)
    metadata.pop("verification_run_id", None)
    metadata.pop("notes", None)
    payload["metadata"] = metadata
    return _sha256(payload)


def cumulative_path_s(points: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Compute cumulative arc length in the provided coordinate basis."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("k_points must be 2D")
    if arr.shape[0] == 0:
        return ()
    step = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    return tuple(float(v) for v in np.concatenate([[0.0], np.cumsum(step)]))


def detect_duplicate_coordinate_groups(
    points: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-10,
) -> tuple[tuple[int, ...], ...]:
    """Group physically identical sampled coordinates using an explicit tolerance."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("k_points must be 2D")
    used: set[int] = set()
    groups: list[tuple[int, ...]] = []
    for i in range(arr.shape[0]):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, arr.shape[0]):
            if j in used:
                continue
            if np.linalg.norm(arr[i] - arr[j]) <= float(tolerance):
                group.append(j)
                used.add(j)
        if len(group) > 1:
            groups.append(tuple(group))
    return tuple(groups)


def physical_k_ids(points: Sequence[Sequence[float]], *, tolerance: float = 1.0e-10) -> tuple[str, ...]:
    """Assign stable physical coordinate ids, merging duplicate coordinates."""
    n = int(np.asarray(points).shape[0])
    ids = [f"pk_{i:04d}" for i in range(n)]
    for group in detect_duplicate_coordinate_groups(points, tolerance=tolerance):
        label = f"pk_{group[0]:04d}"
        for index in group:
            ids[int(index)] = label
    return tuple(ids)


def validate_k_path_spec(spec: KPathSpec, *, expected_nk: int | None = None) -> list[str]:
    """Return validation problems; an empty list means the spec is internally coherent."""
    problems: list[str] = []
    points = np.asarray(spec.k_points, dtype=np.float64)
    if points.ndim != 2:
        problems.append("k_points must be a 2D array")
        return problems
    nk, dim = int(points.shape[0]), int(points.shape[1])
    if spec.Nk != nk:
        problems.append(f"Nk={spec.Nk} does not match len(k_points)={nk}")
    if expected_nk is not None and int(expected_nk) != nk:
        problems.append(f"expected_nk={expected_nk} does not match len(k_points)={nk}")
    if spec.coordinate_dimension != dim:
        problems.append(f"coordinate_dimension={spec.coordinate_dimension} does not match k_points dim={dim}")
    for name, seq in {
        "k_index": spec.k_index,
        "path_s": spec.path_s,
        "segment_id": spec.segment_id,
        "segment_local_t": spec.segment_local_t,
    }.items():
        if len(seq) != nk:
            problems.append(f"{name} length {len(seq)} does not match Nk={nk}")
    if tuple(spec.k_index) != tuple(range(nk)):
        problems.append("k_index must be 0..Nk-1")
    if any(float(spec.path_s[i]) > float(spec.path_s[i + 1]) for i in range(max(0, nk - 1))):
        problems.append("path_s must be monotonic non-decreasing")
    for segment in spec.segments:
        if segment.start_index < 0 or segment.end_index >= nk or segment.start_index > segment.end_index:
            problems.append(f"segment {segment.start_node}->{segment.end_node} has invalid index bounds")
        if segment.number_of_points != segment.end_index - segment.start_index + 1:
            problems.append(f"segment {segment.start_node}->{segment.end_node} number_of_points mismatch")
    expected_points_hash = hash_k_path_points(spec.k_points)
    if spec.k_points_hash and spec.k_points_hash != expected_points_hash:
        problems.append("k_points_hash does not match k_points")
    return problems


def generate_k_points_from_spec(spec: KPathSpec) -> np.ndarray:
    """Return sampled k-points after validating a KPathSpec."""
    problems = validate_k_path_spec(spec)
    if problems:
        raise ValueError("; ".join(problems))
    return np.asarray(spec.k_points, dtype=np.float32)


def _load_spec_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise TypeError(f"k-path sidecar must contain a mapping: {path}")
    return dict(payload)


def load_k_path_spec(path: str | Path) -> KPathSpec:
    """Load a KPathSpec from JSON or YAML sidecar."""
    payload = _load_spec_payload(Path(path))
    metadata_payload = dict(payload.get("metadata") or {})
    return KPathSpec(
        schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        k_path_id=str(payload["k_path_id"]),
        coordinate_system=str(payload["coordinate_system"]),
        coordinate_dimension=int(payload["coordinate_dimension"]),
        reciprocal_basis_convention=str(payload.get("reciprocal_basis_convention", "")),
        nodes=tuple(KPathNode(str(row["label"]), tuple(float(v) for v in row["coordinate"])) for row in payload.get("nodes", [])),
        segments=tuple(
            KPathSegment(
                start_node=str(row["start_node"]),
                end_node=str(row["end_node"]),
                start_index=int(row["start_index"]),
                end_index=int(row["end_index"]),
                number_of_points=int(row["number_of_points"]),
                endpoint_policy=str(row["endpoint_policy"]),
            )
            for row in payload.get("segments", [])
        ),
        k_points=_as_tuple_points(payload["k_points"]),
        k_index=tuple(int(v) for v in payload.get("k_index", range(len(payload["k_points"])))),
        path_s=tuple(float(v) for v in payload.get("path_s", cumulative_path_s(payload["k_points"]))),
        segment_id=tuple(int(v) for v in payload.get("segment_id", [0] * len(payload["k_points"]))),
        segment_local_t=tuple(float(v) for v in payload.get("segment_local_t", [0.0] * len(payload["k_points"]))),
        node_indices={str(k): int(v) for k, v in dict(payload.get("node_indices") or {}).items()},
        duplicate_coordinate_groups=tuple(tuple(int(i) for i in group) for group in payload.get("duplicate_coordinate_groups", [])),
        Nk=int(payload.get("Nk", len(payload["k_points"]))),
        metadata=KPathMetadata(
            source_type=str(metadata_payload.get("source_type", "")),
            source_path=str(metadata_payload.get("source_path", "")),
            source_commit=str(metadata_payload.get("source_commit", "")),
            source_run_id=str(metadata_payload.get("source_run_id", "")),
            source_config_hash=str(metadata_payload.get("source_config_hash", "")),
            verification_status=str(metadata_payload.get("verification_status", "unverified")),
            verification_run_id=str(metadata_payload.get("verification_run_id", "")),
            notes=tuple(str(v) for v in metadata_payload.get("notes", ())),
        ),
        k_points_hash=str(payload.get("k_points_hash", "")),
    )


def save_k_path_spec(spec: KPathSpec, path: str | Path) -> Path:
    """Write a KPathSpec as UTF-8 JSON."""
    problems = validate_k_path_spec(spec)
    if problems:
        raise ValueError("; ".join(problems))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def discover_k_path_sidecar(dataset_path: str | Path) -> Path | None:
    """Discover a dataset-specific or manifest-referenced k-path sidecar."""
    dataset = Path(dataset_path)
    metadata_dir = dataset.parent / "metadata"
    candidates = [
        metadata_dir / f"{dataset.stem}{SIDECAR_SUFFIX}",
        metadata_dir / f"{dataset.stem}{MANIFEST_SUFFIX}",
        dataset.with_name(f"{dataset.stem}{SIDECAR_SUFFIX}"),
        dataset.with_name(f"{dataset.stem}{MANIFEST_SUFFIX}"),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() in {".yaml", ".yml"} and candidate.name.endswith(MANIFEST_SUFFIX):
            payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            sidecar = payload.get("k_path_sidecar") if isinstance(payload, Mapping) else None
            if sidecar:
                sidecar_path = (candidate.parent / str(sidecar)).resolve() if not Path(str(sidecar)).is_absolute() else Path(str(sidecar))
                if sidecar_path.is_file():
                    return sidecar_path
            continue
        return candidate
    return None


def load_dataset_k_path_spec(dataset_path: str | Path) -> KPathSpec | None:
    """Load a sidecar for a dataset when one is discoverable."""
    sidecar = discover_k_path_sidecar(dataset_path)
    if sidecar is None:
        return None
    return load_k_path_spec(sidecar)


def attach_k_path_metadata(dataset_path: str | Path, spec: KPathSpec) -> dict[str, Any]:
    """Return loader-friendly metadata fields for a dataset/spec association."""
    return {
        "dataset_path": str(dataset_path),
        "k_path_available": True,
        "physical_k_coordinates_available": True,
        "k_path_verified": spec.metadata.verification_status in VERIFIED_K_PATH_STATUSES,
        "k_path_id": spec.k_path_id,
        "k_path_hash": hash_k_path_spec(spec),
        "k_points_hash": spec.k_points_hash or hash_k_path_points(spec.k_points),
        "coordinate_system": spec.coordinate_system,
        "coordinate_dimension": spec.coordinate_dimension,
        "segment_count": len(spec.segments),
        "verification_status": spec.metadata.verification_status,
    }


def compare_k_paths(
    left: KPathSpec | Sequence[Sequence[float]],
    right: KPathSpec | Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Compare two sampled k paths without silently converting coordinate systems."""
    left_points = np.asarray(left.k_points if isinstance(left, KPathSpec) else left, dtype=np.float64)
    right_points = np.asarray(right.k_points if isinstance(right, KPathSpec) else right, dtype=np.float64)
    if left_points.shape != right_points.shape:
        return {"same_shape": False, "matches": False, "left_shape": tuple(left_points.shape), "right_shape": tuple(right_points.shape)}
    delta = left_points - right_points
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    return {
        "same_shape": True,
        "matches": bool(max_abs <= float(tolerance)),
        "max_abs_coordinate_delta": max_abs,
        "tolerance": float(tolerance),
        "left_hash": hash_k_path_points(left_points),
        "right_hash": hash_k_path_points(right_points),
    }


def validate_verified_k_path_reference(
    source: KPathSpec,
    reference: KPathSpec,
    *,
    tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Prove a source path inherits a verified physical-k coordinate set.

    Only coordinate verification is inherited; this function never transfers
    dataset provenance or MPB-replay status from the reference dataset.
    """

    if reference.metadata.verification_status not in VERIFIED_K_PATH_STATUSES:
        raise ValueError("The physical-k reference sidecar is not verified.")
    if source.k_path_id != reference.k_path_id:
        raise ValueError("Source and verified reference have different k_path_id values.")
    if source.coordinate_system != reference.coordinate_system:
        raise ValueError("Source and verified reference use different coordinate systems.")
    if source.reciprocal_basis_convention != reference.reciprocal_basis_convention:
        raise ValueError("Source and verified reference use different reciprocal-basis conventions.")
    comparison = compare_k_paths(source, reference, tolerance=tolerance)
    if not comparison["matches"]:
        raise ValueError("Source k-points do not exactly match the verified physical-k reference.")
    source_hash = source.k_points_hash or hash_k_path_points(source.k_points)
    reference_hash = reference.k_points_hash or hash_k_path_points(reference.k_points)
    if source_hash != reference_hash:
        raise ValueError("Source and verified reference have different k_points_hash values.")
    return {
        "coordinate_verification": "inherited_from_exact_verified_reference",
        "source_k_path_id": source.k_path_id,
        "source_k_points_hash": source_hash,
        "source_k_path_status": source.metadata.verification_status,
        "reference_k_path_id": reference.k_path_id,
        "reference_k_points_hash": reference_hash,
        "reference_k_path_status": reference.metadata.verification_status,
        "coordinate_comparison": comparison,
    }


def build_square_gxm_reduced_spec(
    *,
    n_total: int,
    lattice_constant: float = 1.0,
    k_path_id: str = "paper_like_square_gxm_mpb_reduced_v1",
    metadata: KPathMetadata | None = None,
) -> KPathSpec:
    """Build the historical square Γ-X-M-Γ MPB-reduced candidate path."""
    if int(n_total) < 4:
        raise ValueError("n_total must be at least 4")
    m = int(round((int(n_total) + 2) / 3)) if (int(n_total) + 2) % 3 != 0 else (int(n_total) + 2) // 3

    def seg(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
        t = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
        return (1.0 - t)[:, None] * p0[None, :] + t[:, None] * p1[None, :]

    # Historical code produced physical radian coordinates and then multiplied
    # by a/(2*pi) before passing mp.Vector3 to MPB.
    reduced_scale = float(lattice_constant) / (2.0 * np.pi)
    physical_scale = np.pi / float(lattice_constant)
    gamma = np.array([0.0, 0.0], dtype=np.float64)
    x_node = np.array([physical_scale, 0.0], dtype=np.float64) * reduced_scale
    m_node = np.array([physical_scale, physical_scale], dtype=np.float64) * reduced_scale
    k_gx = seg(gamma, x_node, m)
    k_xm = seg(x_node, m_node, m)[1:]
    k_mg = seg(m_node, gamma, m)[1:]
    points = np.vstack([k_gx, k_xm, k_mg])
    if points.shape[0] > int(n_total):
        points = points[: int(n_total)]
    elif points.shape[0] < int(n_total):
        extra = int(n_total) - int(points.shape[0])
        points = np.vstack([points, np.linspace(points[-1], gamma, extra + 2, dtype=np.float64)[1 : extra + 1]])

    segment_bounds = [(0, m - 1), (m - 1, 2 * m - 2), (2 * m - 2, int(n_total) - 1)]
    segment_ids = np.zeros(int(n_total), dtype=np.int64)
    local_t = np.zeros(int(n_total), dtype=np.float64)
    for sid, (start, end) in enumerate(segment_bounds):
        end = min(end, int(n_total) - 1)
        segment_ids[start : end + 1] = sid
        denom = max(1, end - start)
        local_t[start : end + 1] = [(i - start) / denom for i in range(start, end + 1)]

    node_indices = {"Gamma": 0, "X": min(m - 1, int(n_total) - 1), "M": min(2 * m - 2, int(n_total) - 1), "Gamma_end": int(n_total) - 1}
    nodes = (
        KPathNode("Gamma", (0.0, 0.0)),
        KPathNode("X", tuple(float(v) for v in x_node)),
        KPathNode("M", tuple(float(v) for v in m_node)),
        KPathNode("Gamma_end", (0.0, 0.0)),
    )
    segments = (
        KPathSegment("Gamma", "X", segment_bounds[0][0], min(segment_bounds[0][1], int(n_total) - 1), min(m, int(n_total)), "include_start_and_end"),
        KPathSegment("X", "M", segment_bounds[1][0], min(segment_bounds[1][1], int(n_total) - 1), min(m, int(n_total) - segment_bounds[1][0]), "share_start_node_without_duplicate_new_row"),
        KPathSegment("M", "Gamma_end", segment_bounds[2][0], min(segment_bounds[2][1], int(n_total) - 1), min(m, int(n_total) - segment_bounds[2][0]), "share_start_node_without_duplicate_new_row; closed_path_duplicates_gamma"),
    )
    point_tuples = _as_tuple_points(points)
    return KPathSpec(
        schema_version=SCHEMA_VERSION,
        k_path_id=k_path_id,
        coordinate_system="mpb_reduced_reciprocal_coordinates",
        coordinate_dimension=2,
        reciprocal_basis_convention="square lattice MPB reduced coordinates; historical physical radians multiplied by a/(2*pi)",
        nodes=nodes,
        segments=segments,
        k_points=point_tuples,
        k_index=tuple(range(int(n_total))),
        path_s=cumulative_path_s(point_tuples),
        segment_id=tuple(int(v) for v in segment_ids),
        segment_local_t=tuple(float(v) for v in local_t),
        node_indices=node_indices,
        duplicate_coordinate_groups=detect_duplicate_coordinate_groups(point_tuples),
        Nk=int(n_total),
        metadata=metadata or KPathMetadata(),
        k_points_hash=hash_k_path_points(point_tuples),
    )


def expand_physical_holdout_indices(
    holdout_indices: Iterable[int],
    physical_ids: Sequence[str],
) -> tuple[int, ...]:
    """Expand held-out k rows so duplicate physical coordinates move together."""
    ids = tuple(str(v) for v in physical_ids)
    requested = {ids[int(i)] for i in holdout_indices}
    return tuple(i for i, value in enumerate(ids) if value in requested)
