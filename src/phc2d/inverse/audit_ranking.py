"""Paired surrogate-versus-solver ranking for local inverse audit candidates.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau


def paired_candidate_ranking(proxy: Sequence[Mapping[str, Any]],
                             solver: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare complete paired groups, selecting among non-anchor candidates.

    Random baseline is the exact average over the same eligible non-anchor
    candidates. Solver evidence may be coarse sampled; this operation does not
    infer numerical convergence or physical validation.
    """
    def key(row: Mapping[str, Any]) -> tuple[int, str, str]:
        return int(row['source_index']), str(row['label']), str(row['parameter_hash'])

    proxy_by_key = {key(row): row for row in proxy}
    solver_by_key = {key(row): row for row in solver}
    if len(proxy_by_key) != len(proxy) or len(solver_by_key) != len(solver):
        raise ValueError('duplicate candidate identity')
    if not proxy_by_key or proxy_by_key.keys() != solver_by_key.keys():
        raise ValueError('proxy and solver candidate identities differ')
    groups: dict[int, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for identity in sorted(proxy_by_key):
        p, s = proxy_by_key[identity], solver_by_key[identity]
        if p.get('status') != 'surrogate_only' or s.get('status') != 'sampled_solver_only':
            raise ValueError('candidate has invalid proxy or solver status')
        if not np.isfinite([p['predicted_margin'], s['gap']['target_margin']]).all():
            raise ValueError('candidate scores must be finite')
        groups[identity[0]].append((p, s))
    result = []
    for source, entries in sorted(groups.items()):
        anchors = [item for item in entries if item[0]['label'] == 'anchor']
        candidates = [item for item in entries if item[0]['label'] != 'anchor']
        if len(anchors) != 1 or len(candidates) < 2:
            raise ValueError('each source needs one anchor and at least two candidates')
        # Stable ordering from identity sorting defines reproducible tie handling.
        selected = max(candidates, key=lambda item: float(item[0]['predicted_margin']))
        true = np.asarray([float(s['gap']['target_margin']) for _, s in candidates])
        predicted = np.asarray([float(p['predicted_margin']) for p, _ in candidates])
        tau = float(kendalltau(predicted, true).statistic)
        anchor = float(anchors[0][1]['gap']['target_margin'])
        selected_true = float(selected[1]['gap']['target_margin'])
        result.append({'source_index': source, 'candidate_count': len(candidates),
                       'selected_label': selected[0]['label'], 'selected_parameter_hash': selected[0]['parameter_hash'],
                       'anchor_solver_margin': anchor, 'selected_solver_margin': selected_true,
                       'selected_gain': selected_true - anchor,
                       'random_expected_gain': float(true.mean()) - anchor,
                       'kendall_tau': tau if np.isfinite(tau) else None,
                       'kendall_status': 'defined' if np.isfinite(tau) else 'undefined_constant_rank'})
    return result


def summarize_paired_gains(groups: Sequence[Mapping[str, Any]], *, seed: int,
                           bootstrap_samples: int = 10000) -> dict[str, Any]:
    """Bootstrap paired gain differences by source, not by candidate.

    Source units are treated as the sampled population; stratified inference
    requires an explicitly defined weighting policy and is not implied here.
    """
    if len(groups) < 2 or bootstrap_samples < 1:
        raise ValueError('at least two sources and a positive bootstrap count are required')
    ids = [int(row['source_index']) for row in groups]
    if len(set(ids)) != len(ids):
        raise ValueError('source identities must be unique')
    differences = np.asarray([float(row['selected_gain']) - float(row['random_expected_gain'])
                              for row in groups], dtype=np.float64)
    if not np.isfinite(differences).all():
        raise ValueError('paired gains must be finite')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(bootstrap_samples, len(differences)))
    means = differences[indices].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return {'source_count': len(differences), 'selected_minus_random_mean': float(differences.mean()),
            'bootstrap_95_low': float(lo), 'bootstrap_95_high': float(hi),
            'resampling_unit': 'source', 'bootstrap_samples': int(bootstrap_samples), 'seed': int(seed),
            'status': 'descriptive_only'}


def summarize_stratified_paired_gains(
    groups: Sequence[Mapping[str, Any]], *, source_strata: Mapping[int, str],
    expected_counts: Mapping[str, int], stratum_weights: Mapping[str, float],
    seed: int, bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Bootstrap paired gains within frozen strata using explicit weights.

    Every expected source must be present exactly once. The caller must freeze
    the source roster, strata, counts and weights before reading results and
    independently validate candidate identities and numerical gates. This
    summary does not infer population weights, handle missing observations,
    or establish statistical significance or physical acceptance.
    """
    if (not isinstance(source_strata, Mapping) or not isinstance(expected_counts, Mapping)
            or not isinstance(stratum_weights, Mapping) or not expected_counts
            or set(expected_counts) != set(stratum_weights)
            or any(not isinstance(k, str) or not k for k in expected_counts)
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in expected_counts.values())
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1):
        raise ValueError('valid strata, expected counts and bootstrap settings are required')
    try:
        weights = {key: float(stratum_weights[key]) for key in expected_counts}
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid stratum weights') from exc
    if (any(isinstance(stratum_weights[key], bool) for key in weights)
            or not all(np.isfinite(value) and value > 0 for value in weights.values())
            or not np.isclose(sum(weights.values()), 1.0, rtol=0, atol=1e-12)):
        raise ValueError('stratum weights must be positive and sum to one')
    if (not all(isinstance(k, int) and not isinstance(k, bool) and isinstance(v, str)
                for k, v in source_strata.items()) or set(source_strata.values()) != set(expected_counts)):
        raise ValueError('source strata must match frozen strata')
    values: dict[str, list[float]] = {key: [] for key in expected_counts}
    seen: set[int] = set()
    for row in groups:
        source = row.get('source_index')
        if isinstance(source, bool) or not isinstance(source, int) or source in seen or source not in source_strata:
            raise ValueError('group source identities must be unique and frozen')
        try:
            difference = float(row['selected_gain']) - float(row['random_expected_gain'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('invalid paired gain') from exc
        if not np.isfinite(difference):
            raise ValueError('paired gains must be finite')
        seen.add(source)
        values[source_strata[source]].append(difference)
    if seen != set(source_strata) or any(len(values[key]) != expected_counts[key] for key in expected_counts):
        raise ValueError('observed groups do not match the frozen stratum counts')
    rng = np.random.default_rng(seed)
    samples = np.zeros(bootstrap_samples, dtype=np.float64)
    means: dict[str, float] = {}
    for key in sorted(values):
        data = np.asarray(values[key], dtype=np.float64)
        means[key] = float(data.mean())
        indices = rng.integers(0, len(data), size=(bootstrap_samples, len(data)))
        samples += weights[key] * data[indices].mean(axis=1)
    lo, hi = np.quantile(samples, [.025, .975])
    return {'source_count': len(seen), 'stratum_counts': dict(expected_counts),
            'stratum_weights': weights, 'stratum_means': means,
            'selected_minus_random_mean': float(sum(weights[key] * means[key] for key in weights)),
            'bootstrap_95_low': float(lo), 'bootstrap_95_high': float(hi),
            'resampling_unit': 'source_within_stratum', 'bootstrap_samples': bootstrap_samples,
            'seed': seed, 'status': 'descriptive_only'}


def summarize_seed_averaged_stratified_gains(
    groups: Sequence[Mapping[str, Any]], *, source_strata: Mapping[int, str], seeds: Sequence[int],
    expected_counts: Mapping[str, int], stratum_weights: Mapping[str, float],
    bootstrap_seed: int, bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Average frozen model seeds per physical source before stratified bootstrap.

    The complete source-by-seed roster is mandatory. All seeds for one source
    must name the same already-validated physical group digest. Callers must
    independently verify that digest, the numerical gates and the eligible
    candidate sets; this function cannot establish physical acceptance or
    population representativeness.
    """
    if (not isinstance(source_strata, Mapping) or not source_strata
            or any(isinstance(source, bool) or not isinstance(source, int) for source in source_strata)
            or not isinstance(seeds, Sequence)
            or isinstance(seeds, (str, bytes)) or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('valid frozen sources and unique model seeds are required')
    expected = {(source, seed) for source in source_strata for seed in seeds}
    by_source: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    identities: set[tuple[int, int]] = set()
    for row in groups:
        source, model_seed = row.get('source_index'), row.get('seed')
        if (isinstance(source, bool) or not isinstance(source, int)
                or isinstance(model_seed, bool) or not isinstance(model_seed, int)
                or (source, model_seed) not in expected or (source, model_seed) in identities):
            raise ValueError('source and model seed must match the frozen roster exactly once')
        physical = row.get('physical_group_sha256')
        if not isinstance(physical, str) or not re.fullmatch(r'[0-9a-f]{64}', physical):
            raise ValueError('physical group digest is required')
        identities.add((source, model_seed))
        by_source[source].append(row)
    if identities != expected:
        raise ValueError('incomplete source and model seed roster')
    averaged = []
    for source in sorted(by_source):
        rows = by_source[source]
        if len({row['physical_group_sha256'] for row in rows}) != 1:
            raise ValueError('model seeds must share one physical group per source')
        try:
            if any(isinstance(row[key], bool) for row in rows for key in ('selected_gain', 'random_expected_gain')):
                raise ValueError('boolean gain')
            selected = np.asarray([float(row['selected_gain']) for row in rows], dtype=np.float64)
            random = np.asarray([float(row['random_expected_gain']) for row in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('paired gains must be finite numbers') from exc
        if not np.isfinite(selected).all() or not np.isfinite(random).all():
            raise ValueError('paired gains must be finite numbers')
        averaged.append({'source_index': source, 'selected_gain': float(selected.mean()),
                         'random_expected_gain': float(random.mean())})
    result = summarize_stratified_paired_gains(averaged, source_strata=source_strata,
        expected_counts=expected_counts, stratum_weights=stratum_weights,
        seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
    return {**result, 'model_seeds': list(seeds), 'source_seed_count': len(identities),
            'seed_aggregation': 'arithmetic_mean_within_source_before_resampling'}
