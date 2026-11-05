"""Compare paired independent inverse-design endpoints and source-level effects.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


_PURE = re.compile(r'^source_(\d+)_initial_(\d+)_(adam|de|random)/top1$')
_FEEDBACK = re.compile(r'^source_(\d+)_initial_(\d+)_adam_(adam_trajectory|adam_residual|de_residual)/final$')


def analyze_independent(pure: Mapping[str, Any], feedback: Mapping[str, Any]) -> dict[str, Any]:
    """Pair same-source, target and seed outcomes; do not treat repeats as sources."""
    if not pure or set(pure) != set(feedback):
        raise ValueError('independent pure and feedback cell rosters differ')
    cases: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    summaries: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    targets: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for cell in sorted(pure):
        cell_match = re.fullmatch(r'target_(\d+)_seed_(\d+)', cell)
        if not cell_match:
            raise ValueError('independent target or seed identity mismatch')
        for reports, pattern in ((pure, _PURE), (feedback, _FEEDBACK)):
            report = reports[cell]
            if report.get('scope') != 'locked_robust_surrogate_vs_physical':
                raise ValueError('expected locked physical comparison')
            for row in report['roles']:
                match = pattern.fullmatch(row['label'])
                if not match:
                    raise ValueError('independent role identity mismatch')
                source, initial, method = match.groups()
                key = (cell, source, initial)
                if method in cases[key]:
                    raise ValueError('duplicate independent role')
                cases[key][method] = row
                summaries[method].append(row)
                targets[cell_match.group(1)][method].append(row)
    expected = {'adam', 'de', 'random', 'adam_trajectory', 'adam_residual', 'de_residual'}
    if any(set(value) != expected for value in cases.values()):
        raise ValueError('independent paired method roster incomplete')
    def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        stable = [x for x in rows if x['status'] == 'numerically_stable']
        nominal = [float(x['physical_nominal_target_margin']) for x in stable]
        common = [float(x['physical_common_target_margin']) for x in stable]
        if not all(math.isfinite(x) for x in nominal+common):
            raise ValueError('nonfinite physical margin')
        return {'registered':len(rows), 'numerically_stable':len(stable),
            'nominal_target_covered':sum(x > 0 for x in nominal),
            'common_target_covered':sum(x['physical_contains_target'] is True for x in stable),
            'mean_nominal_margin':statistics.mean(nominal) if nominal else None,
            'mean_common_margin':statistics.mean(common) if common else None}
    groups = {method:summarize(rows) for method,rows in sorted(summaries.items())}
    by_target = {target:{method:summarize(rows) for method,rows in sorted(methods.items())}
                 for target,methods in sorted(targets.items())}
    source_rows: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for (_,source,_), methods in cases.items():
        for method,row in methods.items():
            source_rows[source][method].append(row)
    source_groups = {source:{method:summarize(rows) for method,rows in sorted(methods.items())}
                     for source,methods in sorted(source_rows.items())}
    comparisons = {}
    for name, left, right in (('trajectory_minus_adam','adam_trajectory','adam'),
                              ('residual_minus_trajectory','adam_residual','adam_trajectory'),
                              ('residual_adam_minus_adam','adam_residual','adam'),
                              ('residual_de_minus_de','de_residual','de')):
        by_source: dict[str, list[float]] = defaultdict(list)
        for (_, source, _), row in cases.items():
            a,b = row[left],row[right]
            if a['status'] == b['status'] == 'numerically_stable':
                by_source[source].append(float(a['physical_common_target_margin'])-float(b['physical_common_target_margin']))
        source_means = {s:statistics.mean(v) for s,v in sorted(by_source.items())}
        comparisons[name] = {'paired_roles':sum(map(len,by_source.values())), 'source_means':source_means,
                             'mean_of_source_means':statistics.mean(source_means.values()) if source_means else None}
    interactions: dict[str, list[float]] = defaultdict(list)
    for (_,source,_), row in cases.items():
        if all(row[m]['status']=='numerically_stable' for m in ('adam','de','adam_residual','de_residual')):
            margin = lambda m: float(row[m]['physical_common_target_margin'])
            interactions[source].append((margin('adam_residual')-margin('adam'))-(margin('de_residual')-margin('de')))
    interaction_means = {s:statistics.mean(v) for s,v in sorted(interactions.items())}
    return {'scope':'paired_independent_physical_results', 'groups':groups, 'by_target_index':by_target,
            'by_source':source_groups,
            'comparisons':comparisons,
            'gradient_feedback_interaction':{'paired_roles':sum(map(len,interactions.values())),
                'source_means':interaction_means,
                'mean_of_source_means':statistics.mean(interaction_means.values()) if interaction_means else None},
            'independent_source_count':len({key[1] for key in cases}), 'paired_case_count':len(cases)}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.run)
    pure={p.parent.name:json.loads(p.read_bytes()) for p in (root/'two_scenario_results').glob('*/physical_comparison.json')}
    feedback={p.parents[1].name:json.loads(p.read_bytes()) for p in (root/'feedback').glob('*/final_scenario_results/physical_comparison.json')}
    result=analyze_independent(pure,feedback)
    with Path(args.output).open('x',encoding='utf8') as out:
        json.dump(result,out,indent=2,allow_nan=False)


if __name__ == '__main__':
    main()
