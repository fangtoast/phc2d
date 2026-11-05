"""Freeze the final incumbents of a completed feedback campaign.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from phc2d.inverse.feedback_state import restore_feedback


def _bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, allow_nan=False).encode('utf8')


def freeze_final_feedback(arms: Mapping[str, Any], design: Mapping[str, Any], design_lock: Mapping[str, Any],
                          prior_locks: Sequence[Mapping[str, Any]], *, design_sha256: str,
                          arms_sha256: str, prior_lock_sha256: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve each final incumbent to its frozen hard geometry and prediction."""
    if (arms.get('scope') != 'matched_feedback_arms_initialized' or arms.get('completed_rounds') != 3
            or design.get('scope') != 'preregistered_surrogate_design'
            or design_lock.get('manifest_sha256') != design_sha256 or len(prior_locks) != 4
            or len(prior_lock_sha256) != 4 or prior_lock_sha256[0] != arms.get('selection_lock_sha256')
            or prior_lock_sha256[-1] != arms.get('previous_round_lock_sha256')
            or prior_locks[0].get('purpose') != 'feedback_initial_stencil'
            or [lock.get('round_index') for lock in prior_locks[1:]] != [0, 1, 2]):
        raise ValueError('final feedback campaign or round identity mismatch')
    sources = {f"source_{source['source_id']:04d}": source['source_id'] for source in design['sources']}
    entries, donors = [], {}
    for lock in prior_locks:
        for record in lock['records']:
            candidate = record.get('candidate')
            if candidate is not None:
                entries.append(record)
                if candidate.get('bands') is not None:
                    donors.setdefault(candidate['master_grid_sha256'], candidate)
    trials, records, names = [], [], set()
    for arm in arms['arms']:
        name = f"{arm['seed_id']}_{arm['arm']}"
        source_name = arm['seed_id'].split('_initial_', 1)[0]
        if name in names or source_name not in sources:
            raise ValueError('final feedback arm identity mismatch')
        names.add(name)
        trials.append({'trial_id': name, 'source_id': sources[source_name]})
        role = {'trial_id': name, 'label': f'{name}/final', 'role': 'final'}
        if arm['status'] != 'completed':
            records.append({**role, 'candidate': None, 'status': arm['status']})
            continue
        state = restore_feedback(arm['state'])
        if (state.pending is not None or state.remaining_evaluations != 0 or len(state.history) != 3
                or state.incumbent_hash not in arm['physical_observations']
                or arm['physical_observations'][state.incumbent_hash] != state.incumbent_observation):
            raise ValueError('final feedback state or physical incumbent mismatch')
        matching = [record for record in entries if record.get('trial_id') in (arm['seed_id'], name)
                    and record['candidate']['parameter_hash'] == state.incumbent_hash
                    and record.get('point') == list(state.incumbent_parameters)]
        if not matching:
            raise ValueError('final incumbent has no frozen candidate geometry')
        candidate = copy.deepcopy(matching[-1]['candidate'])
        if candidate.get('bands') is None:
            donor = donors.get(candidate['master_grid_sha256'])
            if donor is None:
                raise ValueError('final incumbent surrogate spectrum is missing')
            candidate['bands'] = copy.deepcopy(donor['bands'])
            candidate['predicted_gap'] = copy.deepcopy(donor['predicted_gap'])
        records.append({**role, 'candidate': candidate, 'status': 'selected',
                        'incumbent_parameter_hash': state.incumbent_hash,
                        'physical_history_count': len(state.history)})
    if len(names) != len(arms['arms']) or not records:
        raise ValueError('empty or duplicate final feedback roster')
    synthetic = {'base_config_sha256': design['base_config_sha256'], 'trials': trials,
                 'parent_design_manifest_sha256': design_sha256, 'final_arms_sha256': arms_sha256}
    if 'experiment_id' in design:
        synthetic['experiment_id'] = design['experiment_id']
    lock = {'scope': 'locked_paired_surrogate_campaign', 'purpose': 'feedback_final',
            'manifest_sha256': hashlib.sha256(_bytes(synthetic)).hexdigest(),
            'final_arms_sha256': arms_sha256, 'k_points': design_lock['k_points'], 'records': records}
    return synthetic, lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('arms', 'design-manifest', 'design-lock', 'seed-lock', 'round0-lock', 'round1-lock',
                 'round2-lock', 'output-directory'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    raw_arms, raw_design, raw_design_lock = (Path(getattr(args, name)).read_bytes()
        for name in ('arms', 'design_manifest', 'design_lock'))
    lock_raw = [Path(getattr(args, name)).read_bytes() for name in ('seed_lock', 'round0_lock', 'round1_lock', 'round2_lock')]
    locks = [json.loads(raw) for raw in lock_raw]
    synthetic, lock = freeze_final_feedback(json.loads(raw_arms), json.loads(raw_design), json.loads(raw_design_lock),
        locks, design_sha256=hashlib.sha256(raw_design).hexdigest(), arms_sha256=hashlib.sha256(raw_arms).hexdigest(),
        prior_lock_sha256=[hashlib.sha256(raw).hexdigest() for raw in lock_raw])
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    (output/'manifest.json').write_bytes(_bytes(synthetic))
    (output/'selection.lock.json').write_bytes(_bytes(lock))


if __name__ == '__main__':
    main()
