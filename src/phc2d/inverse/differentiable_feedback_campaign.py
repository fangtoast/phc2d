"""Initialize matched feedback arms from authenticated stencil observations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

from dataclasses import asdict
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from phc2d.inverse.feedback_state import start_feedback


def initialize_feedback_arms(seed_observations: Mapping[str, Any], *, campaign: Mapping[str, Any],
                             target_frequency: float, expected_seed_ids: set[str] | None = None) -> dict[str, Any]:
    """Give all arms the same authenticated initial observations and budget."""
    if (seed_observations.get('scope') != 'feedback_initial_observations'
            or campaign['rounds'] != 3 or campaign['evaluations_per_round'] != 20
            or campaign['max_observation_roles'] != 8
            or campaign['acceptance'] != 'strict_physical_improvement'):
        raise ValueError('feedback campaign contract mismatch')
    if campaign['arms'] != ['adam_trajectory', 'adam_residual', 'de_residual']:
        raise ValueError('feedback arm roster mismatch')
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in seed_observations['records']:
        grouped.setdefault(record['trial_id'], []).append(record)
    if not grouped:
        raise ValueError('empty feedback seed roster')
    if expected_seed_ids is not None and set(grouped) != expected_seed_ids:
        raise ValueError('feedback seed roster differs from the design manifest')
    arms = []
    radius = float(campaign['initial_trust_radius'])
    factor = float(campaign['rejection_radius_factor'])
    if radius <= 0 or factor != .5:
        raise ValueError('invalid feedback trust policy')
    for seed_id, records in sorted(grouped.items()):
        roles = {row['role']: row for row in records}
        if len(records) != 5 or len(roles) != 5 or set(roles) != {f'offset_{i}' for i in range(5)}:
            raise ValueError('feedback stencil roles incomplete or duplicated')
        center = roles['offset_0']
        bank = [copy.deepcopy(roles[f'offset_{i}']['composite']) for i in range(5)
                if roles[f'offset_{i}']['status'] == 'observed']
        if center['status'] == 'observed':
            comparison = center['physical']['provenance']['comparison_contract_sha256']
            for record in records:
                if record['status'] != 'observed':
                    continue
                composite, physical = record['composite'], record['physical']
                row = physical['row']
                edges = composite.get('physical_edges', [])
                if (composite.get('numerical_gate_passed') is not True
                        or composite.get('surrogate_contract_sha256') != seed_observations['surrogate_contract_sha256']
                        or composite.get('comparison_contract_sha256') != comparison
                        or composite.get('parameter_hash') != physical['provenance']['parameter_hash']
                        or composite.get('parameters') != record['point']
                        or physical.get('status') != 'numerically_stable'
                        or not all(row.get(key) is True for key in ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged'))
                        or len(edges) != 2 or not all(math.isfinite(float(x)) for x in edges)
                        or any(abs(float(edges[i]) - float(row[key])) > 1e-8 for i, key in enumerate(
                            ('refined_lower_edge_frequency', 'refined_upper_edge_frequency')))):
                    raise ValueError('feedback seed observation identity mismatch')
        for arm in campaign['arms']:
            item = {'seed_id': seed_id, 'arm': arm, 'method': 'de' if arm == 'de_residual' else 'adam',
                    'apply_correction': arm != 'adam_trajectory', 'observations': copy.deepcopy(bank),
                    'physical_observations': {row['composite']['parameter_hash']: copy.deepcopy(row['physical'])
                                              for row in records if row['status'] == 'observed'},
                    'surrogate_contract_sha256': seed_observations['surrogate_contract_sha256'],
                    'initial_role_statuses': {key: value['status'] for key, value in roles.items()}}
            if center['status'] != 'observed':
                item.update(status='initial_center_unavailable', state=None)
            else:
                observation = center['physical']
                state = start_feedback(incumbent_hash=center['composite']['parameter_hash'],
                    incumbent_parameters=center['point'], incumbent_observation=observation,
                    target_frequency=float(target_frequency), radius=radius,
                    min_radius=radius * factor ** campaign['rounds'], max_radius=radius,
                    candidate_budget=campaign['rounds'], acceptance_policy=campaign['acceptance'])
                item.update(status='ready', state=asdict(state))
            arms.append(item)
    return {'scope': 'matched_feedback_arms_initialized',
            'selection_lock_sha256': seed_observations['selection_lock_sha256'],
            'physical_manifest_sha256': seed_observations['physical_manifest_sha256'], 'arms': arms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('observations', 'campaign', 'base', 'design-manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    observation_raw, campaign_raw, base_raw, manifest_raw = (Path(getattr(args, key)).read_bytes()
                                               for key in ('observations', 'campaign', 'base', 'design_manifest'))
    observations, manifest = json.loads(observation_raw), json.loads(manifest_raw)
    if (observations.get('design_manifest_sha256') != hashlib.sha256(manifest_raw).hexdigest()
            or manifest['base_config_sha256'] != hashlib.sha256(base_raw).hexdigest()):
        raise ValueError('feedback observations or base do not match design manifest')
    seed_ids = {f"source_{source['source_id']:04d}_initial_{item['initial_index']}_adam"
                for source in manifest['sources'] for item in source['initials']}
    result = initialize_feedback_arms(observations, campaign=json.loads(campaign_raw),
                                      target_frequency=json.loads(base_raw)['target_frequency'],
                                      expected_seed_ids=seed_ids)
    result.update(observations_sha256=hashlib.sha256(observation_raw).hexdigest(),
                  campaign_sha256=hashlib.sha256(campaign_raw).hexdigest(),
                  base_config_sha256=hashlib.sha256(base_raw).hexdigest())
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
