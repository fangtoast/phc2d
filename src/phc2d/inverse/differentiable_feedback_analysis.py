"""Analyze paired feedback designs against the locked pure-surrogate baseline.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.inverse.feedback_state import restore_feedback


def analyze_feedback(arms: Mapping[str, Any], baseline: Mapping[str, Any],
                     round_records: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Compare final verified incumbents and prefit errors without physical reranking."""
    if arms.get('scope') != 'matched_feedback_arms_initialized' or arms.get('completed_rounds') != 3:
        raise ValueError('completed feedback campaign required')
    raw = {row['label']: row for row in baseline['roles']}
    if len(raw) != len(baseline['roles']):
        raise ValueError('duplicate pure-surrogate baseline role')
    groups: dict[str, dict[str, Any]] = {}
    for arm in arms['arms']:
        seed, method = arm['seed_id'], arm['arm']
        name = f'{seed}_{method}'
        if method not in ('adam_trajectory', 'adam_residual', 'de_residual') or method in groups.setdefault(seed, {}):
            raise ValueError('feedback arm identity duplicated or unsupported')
        item = {'status': arm['status'], 'final_margin': None, 'rounds': []}
        if arm['status'] == 'completed':
            state = restore_feedback(arm['state'])
            records = round_records.get(name, [])
            if state.remaining_evaluations != 0 or len(state.history) != 3 or len(records) != 3:
                raise ValueError('feedback arm has incomplete round evidence')
            physical = state.incumbent_observation
            if physical != arm['physical_observations'][state.incumbent_hash]:
                raise ValueError('feedback final physical incumbent differs')
            row = physical['row']
            target = state.target_frequency
            lower, upper = float(row['refined_lower_edge_frequency']), float(row['refined_upper_edge_frequency'])
            if not np.isfinite([lower, upper]).all() or not all(row.get(key) is True for key in
                    ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged')):
                raise ValueError('feedback final incumbent did not pass numerical gates')
            item.update(final_margin=float(2*min(target-lower, upper-target)/target),
                        final_edges=[lower, upper], incumbent_parameter_hash=state.incumbent_hash)
            for index, record in enumerate(records):
                completed, request = record['completion'], record['request']
                frozen, physical_row = request['frozen'], record['composite']['physical_edges']
                state_after = restore_feedback(completed['state'])
                if (len(state_after.history) != index + 1 or list(state_after.history) != list(state.history[:index+1])
                        or completed['proposal_sha256'] != frozen['proposal_sha256']):
                    raise ValueError('feedback round state or prediction identity mismatch')
                pred = frozen['prefit_prediction']
                actual = np.asarray(physical_row, dtype=float)
                original, corrected = (np.asarray(pred[key], dtype=float) for key in ('soft_smooth_edges', 'corrected_smooth_edges'))
                if (actual.shape != (2,) or original.shape != (2,) or corrected.shape != (2,)
                        or not all(np.isfinite(x).all() for x in (actual, original, corrected))
                        or not np.allclose(corrected-actual, completed['prefit_edge_error'], rtol=0, atol=1e-8)):
                    raise ValueError('feedback prefit error does not reproduce')
                item['rounds'].append({'round': index, 'accepted': completed['accepted'], 'cached': completed['cached'],
                    'raw_edge_error': (original-actual).tolist(), 'corrected_edge_error': (corrected-actual).tolist()})
        groups[seed][method] = item
    rows = []
    for seed, values in sorted(groups.items()):
        if set(values) != {'adam_trajectory', 'adam_residual', 'de_residual'} or not seed.endswith('_adam'):
            raise ValueError('feedback paired arm roster incomplete')
        pure_adam = raw.get(seed + '/top1')
        pure_de = raw.get(seed[:-5] + '_de/top1')
        if pure_adam is None or pure_de is None:
            raise ValueError('paired pure-surrogate baseline missing')
        def baseline_margin(row: Mapping[str, Any]) -> float | None:
            return float(row['physical_refined_target_margin']) if row['status'] == 'numerically_stable' else None
        adam, de = baseline_margin(pure_adam), baseline_margin(pure_de)
        trajectory = values['adam_trajectory']['final_margin']
        corrected_adam, corrected_de = values['adam_residual']['final_margin'], values['de_residual']['final_margin']
        interaction = ((corrected_adam-adam)-(corrected_de-de)
                       if all(x is not None for x in (adam, de, corrected_adam, corrected_de)) else None)
        rows.append({'seed_id': seed, 'pure_adam': adam, 'pure_de': de,
                     'adam_trajectory': trajectory, 'adam_residual': corrected_adam,
                     'de_residual': corrected_de, 'interaction': interaction, 'arms': values})
    summary = {}
    for method in ('pure_adam', 'pure_de', 'adam_trajectory', 'adam_residual', 'de_residual'):
        available = [row[method] for row in rows if row[method] is not None]
        summary[method] = {'registered': len(rows), 'numerically_valid': len(available),
                           'target_covered': sum(value > 0 for value in available),
                           'mean_margin': float(np.mean(available)) if available else None,
                           'median_margin': float(np.median(available)) if available else None}
    paired = {}
    for name, comparison, reference in (('trajectory_vs_pure_adam', 'adam_trajectory', 'pure_adam'),
                                        ('residual_vs_trajectory', 'adam_residual', 'adam_trajectory'),
                                        ('residual_adam_vs_pure', 'adam_residual', 'pure_adam'),
                                        ('residual_de_vs_pure', 'de_residual', 'pure_de')):
        values = [row[comparison]-row[reference] for row in rows
                  if row[comparison] is not None and row[reference] is not None]
        paired[name] = {'count': len(values), 'mean': float(np.mean(values)) if values else None,
                        'median': float(np.median(values)) if values else None}
    interaction_by_source: dict[str, list[float]] = {}
    for row in rows:
        if row['interaction'] is not None:
            interaction_by_source.setdefault(row['seed_id'].split('_initial_', 1)[0], []).append(row['interaction'])
    source_means = {key: float(np.mean(values)) for key, values in interaction_by_source.items()}
    errors = {}
    for method in ('adam_trajectory', 'adam_residual', 'de_residual'):
        rounds = [round_row for row in rows for round_row in row['arms'][method]['rounds']]
        errors[method] = {'observation_count': len(rounds),
            'raw_edge_mae': float(np.mean([abs(value) for item in rounds for value in item['raw_edge_error']])) if rounds else None,
            'corrected_edge_mae': float(np.mean([abs(value) for item in rounds for value in item['corrected_edge_error']])) if rounds else None}
    return {'scope': 'paired_development_feedback_comparison', 'seeds': rows,
            'registered_seed_count': len(rows),
            'complete_2x2_count': sum(row['interaction'] is not None for row in rows), 'method_summary': summary,
            'paired_changes': paired, 'prefit_errors': errors, 'interaction_by_source': source_means,
            'interaction_mean_across_sources': float(np.mean(list(source_means.values()))) if source_means else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('arms', 'baseline', 'round0', 'round1', 'round2', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    arms_raw, baseline_raw = Path(args.arms).read_bytes(), Path(args.baseline).read_bytes()
    arms, records = json.loads(arms_raw), {}
    roots = [Path(args.round0), Path(args.round1), Path(args.round2)]
    for arm in arms['arms']:
        if arm['status'] != 'completed':
            continue
        name = f"{arm['seed_id']}_{arm['arm']}"
        records[name] = []
        for root in roots:
            request = json.loads((root/name/'request.json').read_bytes())
            result = json.loads((root/name/'result.json').read_bytes())
            records[name].append({**result, 'request': request})
    result = analyze_feedback(arms, json.loads(baseline_raw), records)
    result.update(arms_sha256=hashlib.sha256(arms_raw).hexdigest(),
                  baseline_sha256=hashlib.sha256(baseline_raw).hexdigest())
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
