"""Candidate batches for generative inverse design.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class CandidateBatch:
    """Generated candidate structures for one inverse-design condition."""

    target_id: str
    condition: Mapping[str, Any]
    rho_candidates: np.ndarray | None = None
    input_repr: str = ""
    metadata: list[dict[str, Any]] = field(default_factory=list)
    grid_candidates: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id must be a non-empty string")
        if not isinstance(self.condition, Mapping):
            raise TypeError("condition must be a mapping")
        if not self.input_repr:
            raise ValueError("input_repr must be provided")
        has_rho = self.rho_candidates is not None
        has_grid = self.grid_candidates is not None
        if has_rho == has_grid:
            raise ValueError("provide exactly one of rho_candidates or grid_candidates")

        arr = np.asarray(self.rho_candidates if has_rho else self.grid_candidates, dtype=np.float32)
        if arr.ndim < 2:
            raise ValueError("candidate arrays must include a candidate axis and feature axes")
        if arr.shape[0] <= 0:
            raise ValueError("candidate arrays must contain at least one candidate")
        if not np.isfinite(arr).all():
            raise ValueError("candidate arrays must contain only finite values")

        if has_rho:
            object.__setattr__(self, "rho_candidates", arr)
        else:
            object.__setattr__(self, "grid_candidates", arr)

        if self.metadata:
            if len(self.metadata) != int(arr.shape[0]):
                raise ValueError("metadata length must match the number of candidates")
            object.__setattr__(self, "metadata", [dict(item) for item in self.metadata])
        else:
            object.__setattr__(
                self,
                "metadata",
                [{"candidate_index": int(index)} for index in range(int(arr.shape[0]))],
            )
        object.__setattr__(self, "condition", dict(self.condition))

    @property
    def n_candidates(self) -> int:
        return int(self.candidates.shape[0])

    @property
    def candidates(self) -> np.ndarray:
        """Return the active candidate tensor, independent of representation."""

        if self.rho_candidates is not None:
            return self.rho_candidates
        if self.grid_candidates is not None:
            return self.grid_candidates
        raise RuntimeError("CandidateBatch invariant broken: no candidates available")

    @property
    def candidate_kind(self) -> str:
        """Return whether this batch stores rho vectors or grids."""

        return "rho_candidates" if self.rho_candidates is not None else "grid_candidates"


class GenerativeInverseAdapter(Protocol):
    """Minimal sampling contract for conditional VAE/GAN/diffusion adapters."""

    name: str
    output_repr: str

    def sample(self, condition: Any, *, n_candidates: int, seed: int) -> CandidateBatch:
        """Sample reproducible candidates for one target condition."""


def diversity_hamming(rho_candidates: np.ndarray, *, threshold: float = 0.5) -> float:
    """Mean pairwise Hamming distance for binary/soft candidate arrays."""

    arr = np.asarray(rho_candidates)
    if arr.shape[0] < 2:
        return 0.0
    binary = arr.reshape(arr.shape[0], -1) >= float(threshold)
    total = 0.0
    count = 0
    for i in range(binary.shape[0]):
        for j in range(i + 1, binary.shape[0]):
            total += float(np.mean(binary[i] != binary[j]))
            count += 1
    return total / float(max(count, 1))


def nearest_neighbor_hamming(rho_candidates: np.ndarray, *, threshold: float = 0.5) -> float:
    """Mean nearest-neighbor Hamming distance in a single target group."""

    return float(nearest_neighbor_hamming_stats(rho_candidates, threshold=threshold)["nearest_neighbor_hamming_mean"])


def nearest_neighbor_hamming_p10(rho_candidates: np.ndarray, *, threshold: float = 0.5, percentile: float = 10.0) -> float:
    """p10 nearest-neighbor Hamming distance; hard collision early-warning signal."""

    _ = percentile
    return float(nearest_neighbor_hamming_stats(rho_candidates, threshold=threshold)["nearest_neighbor_hamming_p10"])


def nearest_neighbor_hamming_stats(rho_candidates: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    """Return hard duplicate and nearest-neighbor diagnostics for one target."""

    arr = np.asarray(rho_candidates)
    n = int(arr.shape[0])
    if n == 0:
        return {
            "nearest_neighbor_hamming_min": 0.0,
            "nearest_neighbor_hamming_p10": 0.0,
            "nearest_neighbor_hamming_mean": 0.0,
            "duplicate_cluster_count": 0.0,
            "largest_duplicate_cluster_fraction": 0.0,
        }
    binary = (arr.reshape(n, -1) >= float(threshold)).astype(np.uint8)
    _, counts = np.unique(binary, axis=0, return_counts=True)
    duplicate_counts = counts[counts > 1]
    duplicate_cluster_count = int(duplicate_counts.size)
    largest_duplicate_cluster_fraction = float(duplicate_counts.max() / n) if duplicate_counts.size else 0.0
    if n < 2:
        return {
            "nearest_neighbor_hamming_min": 0.0,
            "nearest_neighbor_hamming_p10": 0.0,
            "nearest_neighbor_hamming_mean": 0.0,
            "duplicate_cluster_count": float(duplicate_cluster_count),
            "largest_duplicate_cluster_fraction": largest_duplicate_cluster_fraction,
        }
    all_min: list[float] = []
    for i in range(n):
        ref = binary[i]
        d = np.mean(binary != ref, axis=1)
        d[i] = 1.0
        all_min.append(float(d.min()))
    nearest = np.asarray(all_min, dtype=np.float32)
    return {
        "nearest_neighbor_hamming_min": float(np.min(nearest)),
        "nearest_neighbor_hamming_p10": float(np.percentile(nearest, 10.0)),
        "nearest_neighbor_hamming_mean": float(np.mean(nearest)),
        "duplicate_cluster_count": float(duplicate_cluster_count),
        "largest_duplicate_cluster_fraction": largest_duplicate_cluster_fraction,
    }


def separated_candidate_indices(
    rho_candidates: np.ndarray,
    *,
    threshold: float = 0.5,
    deduplicate_binary: bool = False,
    min_hamming: float = 0.0,
) -> np.ndarray:
    """Greedily keep binary-unique candidates separated by a minimum Hamming distance."""

    arr = np.asarray(rho_candidates)
    n = int(arr.shape[0])
    if n == 0:
        return np.asarray([], dtype=np.int64)
    if not deduplicate_binary and float(min_hamming) <= 0.0:
        return np.arange(n, dtype=np.int64)

    binary = (arr.reshape(n, -1) >= float(threshold)).astype(np.uint8)
    selected: list[int] = []
    seen: set[bytes] = set()
    min_distance = max(float(min_hamming), 0.0)
    for idx in range(n):
        row = binary[idx]
        key = row.tobytes()
        if deduplicate_binary and key in seen:
            continue
        if selected and min_distance > 0.0:
            distances = np.mean(binary[np.asarray(selected, dtype=np.int64)] != row, axis=1)
            if float(np.min(distances)) + 1.0e-12 < min_distance:
                continue
        selected.append(idx)
        seen.add(key)

    if not selected:
        selected = [0]
    return np.asarray(selected, dtype=np.int64)


def unique_rate(rho_candidates: np.ndarray, *, threshold: float = 0.5) -> float:
    """Fraction of unique thresholded candidates."""

    arr = np.asarray(rho_candidates)
    if arr.shape[0] == 0:
        return 0.0
    binary = (arr.reshape(arr.shape[0], -1) >= float(threshold)).astype(np.uint8)
    unique = np.unique(binary, axis=0)
    return float(unique.shape[0] / arr.shape[0])


def validity_rate(
    candidates: np.ndarray,
    *,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float:
    """Fraction of candidates with finite values inside the configured range."""

    arr = np.asarray(candidates, dtype=np.float32)
    if arr.shape[0] == 0:
        return 0.0
    flat = arr.reshape(arr.shape[0], -1)
    finite = np.isfinite(flat).all(axis=1)
    bounded = (flat >= float(min_value)).all(axis=1) & (flat <= float(max_value)).all(axis=1)
    return float(np.mean(finite & bounded))


def fill_fraction(candidates: np.ndarray, *, threshold: float = 0.5) -> float:
    """Mean active-material fraction after thresholding candidate tensors."""

    arr = np.asarray(candidates, dtype=np.float32)
    if arr.shape[0] == 0:
        return 0.0
    return float(np.mean(arr.reshape(arr.shape[0], -1) >= float(threshold)))


def candidate_batch_metrics(
    batch: CandidateBatch,
    *,
    threshold: float = 0.5,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> dict[str, float]:
    """Return protocol-level candidate metrics before surrogate or replay stages."""

    arr = batch.candidates
    nn_stats = nearest_neighbor_hamming_stats(arr, threshold=threshold)
    return {
        "n_candidates": float(batch.n_candidates),
        "validity_rate": validity_rate(arr, min_value=min_value, max_value=max_value),
        "unique_rate": unique_rate(arr, threshold=threshold),
        "diversity_hamming": diversity_hamming(arr, threshold=threshold),
        "nearest_neighbor_hamming": nn_stats["nearest_neighbor_hamming_mean"],
        **nn_stats,
        "fill_fraction": fill_fraction(arr, threshold=threshold),
    }


__all__ = [
    "CandidateBatch",
    "GenerativeInverseAdapter",
    "candidate_batch_metrics",
    "diversity_hamming",
    "fill_fraction",
    "nearest_neighbor_hamming",
    "nearest_neighbor_hamming_p10",
    "nearest_neighbor_hamming_stats",
    "separated_candidate_indices",
    "unique_rate",
    "validity_rate",
]
