"""Summarize paired manufacturing-scenario inverse designs.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def analyze_robust_comparison(pure: Mapping[str, Any], corrected: Mapping[str, Any]) -> dict[str, Any]:
    """Report physical coverage and paired differences without replacing missing results."""
    routes: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    modes = ('nominal', 'mean', 'worst')
    for route, report in (('pure', pure), ('corrected', corrected)):
        if report.get('scope') != 'locked_robust_surrogate_vs_physical':
            raise ValueError('expected locked robust physical comparison')
        rows: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in report['roles']:
            label = row['label']
            if not label.endswith('/top1'):
                continue
            trial = label[:-len('/top1')]
            if route == 'corrected':
                if not trial.endswith('_corrected'):
                    raise ValueError('corrected role identity mismatch')
                trial = trial[:-len('_corrected')]
            prefix, mode = trial.rsplit('_', 1)
            if mode not in modes or (prefix, mode) in rows:
                raise ValueError('robust endpoint roster invalid')
            rows[prefix, mode] = row
        if not rows:
            raise ValueError('robust endpoint roster empty')
        routes[route] = rows
    if set(routes['pure']) != set(routes['corrected']):
        raise ValueError('pure and corrected endpoint rosters differ')
    groups, correction, against_nominal = {}, {}, {}
    for route, rows in routes.items():
        groups[route], against_nominal[route] = {}, {}
        for mode in modes:
            registered = [row for (prefix, m), row in rows.items() if m == mode]
            valid = [row for row in registered if row['status'] == 'numerically_stable']
            common = [float(row['physical_common_target_margin']) for row in valid]
            nominal = [float(row['physical_nominal_target_margin']) for row in valid]
            if not np.isfinite(common + nominal).all():
                raise ValueError('nonfinite robust physical margin')
            groups[route][mode] = {'registered': len(registered), 'numerically_valid': len(valid),
                'common_target_covered': sum(row['physical_contains_target'] is True for row in valid),
                'nominal_target_covered': sum(value > 0 for value in nominal),
                'common_width_positive': sum(row['physical_common_edges'][1] > row['physical_common_edges'][0]
                                             for row in valid),
                'mean_common_margin': float(np.mean(common)) if common else None,
                'mean_nominal_margin': float(np.mean(nominal)) if nominal else None}
            if mode != 'nominal':
                pairs = [(row, rows.get((prefix, 'nominal'))) for (prefix, m), row in rows.items() if m == mode]
                pairs = [(x, y) for x, y in pairs if x['status'] == 'numerically_stable'
                         and y is not None and y['status'] == 'numerically_stable']
                against_nominal[route][mode] = {'count': len(pairs),
                    'mean_common_change': float(np.mean([x['physical_common_target_margin']-y['physical_common_target_margin']
                                                        for x, y in pairs])) if pairs else None,
                    'mean_nominal_change': float(np.mean([x['physical_nominal_target_margin']-y['physical_nominal_target_margin']
                                                         for x, y in pairs])) if pairs else None}
    for mode in modes:
        paired = [(routes['corrected'][key], row) for key, row in routes['pure'].items() if key[1] == mode
                  and row['status'] == 'numerically_stable' and routes['corrected'][key]['status'] == 'numerically_stable']
        changes = [x['physical_common_target_margin']-y['physical_common_target_margin'] for x, y in paired]
        correction[mode] = {'count': len(changes), 'mean_common_change': float(np.mean(changes)) if changes else None,
                            'median_common_change': float(np.median(changes)) if changes else None,
                            'better': sum(x > 0 for x in changes), 'worse': sum(x < 0 for x in changes)}
    return {'scope': 'paired_development_manufacturing_comparison', 'groups': groups,
            'corrected_minus_pure': correction, 'mode_minus_nominal': against_nominal}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pure', 'corrected', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    pure_raw, corrected_raw = Path(args.pure).read_bytes(), Path(args.corrected).read_bytes()
    report = analyze_robust_comparison(json.loads(pure_raw), json.loads(corrected_raw))
    report.update(pure_sha256=hashlib.sha256(pure_raw).hexdigest(),
                  corrected_sha256=hashlib.sha256(corrected_raw).hexdigest())
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
