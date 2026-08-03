"""Surrogate ranking and top-M selection for inverse candidates.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from phc2d.inverse.candidates import CandidateBatch, candidate_batch_metrics, diversity_hamming
from phc2d.inverse.morphology import binary_morphology_metrics, candidate_morphology_metrics, morphology_score


class SurrogatePredictor(Protocol):
    """Minimal prediction interface used by candidate ranking."""

    def predict(self, candidates: Any) -> np.ndarray:
        """Return surrogate predictions for all candidates."""


@dataclass(frozen=True)
class SurrogateRankingResult:
    """Ranked candidate indices and scalar metrics from surrogate-side screening."""

    ranked_indices: np.ndarray
    selected_indices: np.ndarray
    surrogate_errors: np.ndarray
    morphology_scores: np.ndarray
    ranking_scores: np.ndarray
    metrics: dict[str, float]


def mean_relative_error(prediction: Any, target: Any, *, epsilon: float = 1.0e-8) -> np.ndarray:
    """Candidate-wise mean relative error with target broadcasting."""

    pred = np.asarray(prediction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if pred.ndim == 0:
        raise ValueError("prediction must include a candidate axis")
    try:
        tgt_broadcast = np.broadcast_to(tgt, pred.shape)
    except ValueError as exc:
        raise ValueError(f"target shape {tgt.shape} cannot broadcast to prediction shape {pred.shape}") from exc
    denom = np.maximum(np.abs(tgt_broadcast), float(epsilon))
    flat = np.abs(pred - tgt_broadcast) / denom
    return flat.reshape(pred.shape[0], -1).mean(axis=1)


def _predict_surrogate(surrogate: SurrogatePredictor | Any, candidates: np.ndarray) -> np.ndarray:
    if hasattr(surrogate, "predict"):
        return np.asarray(surrogate.predict(candidates), dtype=np.float32)
    if callable(surrogate):
        return np.asarray(surrogate(candidates), dtype=np.float32)
    raise TypeError("surrogate must define predict(candidates) or be callable")


def _candidate_hamming(a: np.ndarray, b: np.ndarray, *, threshold: float) -> float:
    af = np.asarray(a).reshape(-1) >= float(threshold)
    bf = np.asarray(b).reshape(-1) >= float(threshold)
    if af.shape != bf.shape:
        raise ValueError("candidate shapes must match for diversity-aware selection")
    return float(np.mean(af != bf))


def _select_top_m(
    candidates: np.ndarray,
    ranking_scores: np.ndarray,
    *,
    top_m: int,
    threshold: float,
    diversity_weight: float,
) -> np.ndarray:
    if top_m <= 0:
        raise ValueError("top_m must be positive")
    n = int(candidates.shape[0])
    if n == 0:
        return np.asarray([], dtype=np.int64)
    target_m = min(int(top_m), n)
    selected: list[int] = []
    remaining = set(range(n))
    while len(selected) < target_m:
        best_index: int | None = None
        best_score: float | None = None
        for idx in sorted(remaining):
            diversity_bonus = 0.0
            if selected and diversity_weight:
                diversity_bonus = min(
                    _candidate_hamming(candidates[idx], candidates[j], threshold=threshold) for j in selected
                )
            adjusted = float(ranking_scores[idx]) - float(diversity_weight) * diversity_bonus
            if best_score is None or adjusted < best_score:
                best_index = idx
                best_score = adjusted
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
    return np.asarray(selected, dtype=np.int64)


def rank_candidates(
    batch: CandidateBatch,
    surrogate: SurrogatePredictor | Any,
    target: Any,
    *,
    top_m: int,
    threshold: float = 0.5,
    morphology_weight: float = 0.0,
    diversity_weight: float = 0.0,
) -> SurrogateRankingResult:
    """Rank candidates by surrogate error, optional morphology, and diversity-aware top-M."""

    start = perf_counter()
    candidates = batch.candidates
    predictions = _predict_surrogate(surrogate, candidates)
    errors = mean_relative_error(predictions, target)
    morph_scores = np.asarray(
        [morphology_score(binary_morphology_metrics(item, threshold=threshold)) for item in candidates],
        dtype=np.float32,
    )
    ranking_scores = errors - float(morphology_weight) * morph_scores
    ranked = np.argsort(ranking_scores, kind="stable").astype(np.int64)
    selected = _select_top_m(
        candidates,
        ranking_scores,
        top_m=top_m,
        threshold=threshold,
        diversity_weight=diversity_weight,
    )
    elapsed = perf_counter() - start
    best = float(errors[ranked[0]]) if ranked.size else 0.0
    metrics = {
        **candidate_batch_metrics(batch, threshold=threshold),
        **candidate_morphology_metrics(candidates, threshold=threshold),
        "best_of_k_surrogate_mre": best,
        "surrogate_ranking_time_s": float(elapsed),
        "inference_time_per_sample_ms": float((elapsed / max(batch.n_candidates, 1)) * 1000.0),
        "selected_count": float(selected.shape[0]),
        "selected_diversity_hamming": diversity_hamming(candidates[selected], threshold=threshold)
        if selected.shape[0] > 1
        else 0.0,
    }
    return SurrogateRankingResult(
        ranked_indices=ranked,
        selected_indices=selected,
        surrogate_errors=errors.astype(np.float32),
        morphology_scores=morph_scores,
        ranking_scores=ranking_scores.astype(np.float32),
        metrics=metrics,
    )


__all__ = [
    "SurrogatePredictor",
    "SurrogateRankingResult",
    "mean_relative_error",
    "rank_candidates",
]
