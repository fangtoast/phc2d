"""Create paired multi-target, multi-seed independent inverse campaigns.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.differentiable_campaign import build_campaign
from phc2d.inverse.independent_sources import validate_frozen_method_lock


def _bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, allow_nan=False).encode('utf8')


def build_independent_matrix(roster: Mapping[str, Any], sources: Mapping[int, Any], *,
                             base: Mapping[str, Any], settings: Mapping[str, Any],
                             roster_sha256: str, method_lock_sha256: str) -> list[dict[str, Any]]:
    """Create design-only configurations without reading test band labels."""
    if (settings.get('experiment_id') != 'EXP-INV-016'
            or roster.get('scope') != 'heldout_inverse_source_selection_no_band_labels'
            or roster.get('method_lock_sha256') != method_lock_sha256
            or settings.get('method_lock_sha256') != method_lock_sha256
            or settings.get('source_roster_sha256') != roster_sha256):
        raise ValueError('independent campaign source or method identity mismatch')
    targets, seeds = settings['target_frequencies'], settings['optimization_seeds']
    if (not targets or len(targets) != len(set(targets)) or not all(math.isfinite(x) and x > 0 for x in targets)
            or not seeds or len(seeds) != len(set(seeds))
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)):
        raise ValueError('invalid independent targets or optimization seeds')
    source_entries = [{'id': row['id'], 'sha256': row['sha256']} for row in roster['sources']]
    if (set(sources) != {row['id'] for row in source_entries} or len(sources) != len(source_entries)
            or any(row['split'] != 'test' for row in roster['sources'])):
        raise ValueError('independent source roster is incomplete or not test-only')
    matrix = []
    for target_index, target in enumerate(targets):
        for seed in seeds:
            resolved_base = {**dict(base), 'experiment_id': settings['experiment_id'], 'target_frequency': float(target)}
            base_raw = _bytes(resolved_base)
            campaign = {'experiment_id': settings['experiment_id'], 'sources': source_entries,
                'methods': settings['methods'], 'evaluations': settings['evaluations'], 'seed': seed,
                'initial_options': settings['initial_options'],
                'diagnostic_evaluation_counts': settings['diagnostic_evaluation_counts'],
                'base_config_sha256': hashlib.sha256(base_raw).hexdigest(),
                'method_lock_sha256': method_lock_sha256, 'source_roster_sha256': roster_sha256,
                'target_index': target_index, 'target_frequency': float(target)}
            campaign_raw = _bytes(campaign)
            design = build_campaign(campaign, resolved_base, sources)
            design.update(campaign_sha256=hashlib.sha256(campaign_raw).hexdigest(),
                          base_config_sha256=campaign['base_config_sha256'],
                          method_lock_sha256=method_lock_sha256, source_roster_sha256=roster_sha256,
                          target_index=target_index, target_frequency=float(target), optimization_seed=seed)
            matrix.append({'target_index': target_index, 'target_frequency': float(target), 'seed': seed,
                           'base': resolved_base, 'campaign': campaign, 'manifest': design})
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('settings', 'roster', 'base', 'method-lock', 'source-directory', 'output-directory'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    settings_raw, roster_raw, base_raw, method_raw = (Path(getattr(args, key)).read_bytes()
        for key in ('settings', 'roster', 'base', 'method_lock'))
    settings, roster = json.loads(settings_raw), json.loads(roster_raw)
    if settings['base_sha256'] != hashlib.sha256(base_raw).hexdigest():
        raise ValueError('independent base model configuration differs')
    validate_frozen_method_lock(method_raw, settings['method_lock_sha256'])
    sources = {}
    for row in roster['sources']:
        raw = (Path(args.source_directory)/f"source_{row['id']:04d}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('independent source file identity mismatch')
        source = FullCellStructureSpec.from_mapping(json.loads(raw))
        if source.parameter_hash != row['parameter_hash']:
            raise ValueError('independent source parameter identity mismatch')
        sources[row['id']] = source
    matrix = build_independent_matrix(roster, sources, base=json.loads(base_raw), settings=settings,
        roster_sha256=hashlib.sha256(roster_raw).hexdigest(), method_lock_sha256=hashlib.sha256(method_raw).hexdigest())
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    index = []
    for entry in matrix:
        directory = output/f"target_{entry['target_index']:02d}_seed_{entry['seed']}"
        directory.mkdir()
        for key in ('base', 'campaign', 'manifest'):
            (directory/f'{key}.json').write_bytes(_bytes(entry[key]))
        index.append({key: entry[key] for key in ('target_index', 'target_frequency', 'seed')})
    (output/'matrix.json').write_bytes(_bytes({'scope': 'preregistered_independent_design_matrix',
        'settings_sha256': hashlib.sha256(settings_raw).hexdigest(), 'cells': index}))


if __name__ == '__main__':
    main()
