"""Inverse-design contracts and adapters.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.inverse.config import DispersionInverseConfig, GenerativeInverseConfig
from phc2d.inverse.candidates import (
    CandidateBatch,
    GenerativeInverseAdapter,
    candidate_batch_metrics,
    diversity_hamming,
    fill_fraction,
    nearest_neighbor_hamming,
    nearest_neighbor_hamming_p10,
    nearest_neighbor_hamming_stats,
    separated_candidate_indices,
    unique_rate,
    validity_rate,
)
from phc2d.inverse.morphology import (
    binary_morphology_metrics,
    candidate_morphology_metrics,
    morphology_score,
)
from phc2d.inverse.selection import (
    SurrogatePredictor,
    SurrogateRankingResult,
    mean_relative_error,
    rank_candidates,
)
from phc2d.inverse.runner import (
    run_inverse_eval,
    run_inverse_objective_train,
    run_inverse_rerank,
    run_inverse_sample,
    run_inverse_train,
)
from phc2d.inverse.diagnostics import run_inverse_design_diagnostic, run_inverse_diagnose
from phc2d.inverse.benchmark import run_a2_benchmark

__all__ = [
    "CandidateBatch",
    "DispersionInverseConfig",
    "GenerativeInverseAdapter",
    "GenerativeInverseConfig",
    "SurrogatePredictor",
    "SurrogateRankingResult",
    "binary_morphology_metrics",
    "candidate_batch_metrics",
    "candidate_morphology_metrics",
    "diversity_hamming",
    "fill_fraction",
    "mean_relative_error",
    "morphology_score",
    "nearest_neighbor_hamming",
    "nearest_neighbor_hamming_p10",
    "nearest_neighbor_hamming_stats",
    "separated_candidate_indices",
    "rank_candidates",
    "unique_rate",
    "validity_rate",
    "run_inverse_eval",
    "run_inverse_design_diagnostic",
    "run_inverse_diagnose",
    "run_inverse_objective_train",
    "run_inverse_rerank",
    "run_inverse_sample",
    "run_inverse_train",
    "run_a2_benchmark",
]
