"""Freeze a feedback round and prepare its deduplicated MPB request.

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

from phc2d.inverse.differentiable_campaign_handoff import build_campaign_request
from phc2d.inverse.differentiable_feedback_round import freeze_round
from phc2d.inverse.feedback_state import restore_feedback


def _serialized(value: Any) -> bytes:
    return json.dumps(value, indent=2, allow_nan=False).encode('utf8')


def validate_round_summaries(summaries: list[Mapping[str, Any]], *, arms: Mapping[str, Any],
                             design_manifest: Mapping[str, Any], arms_sha256: str,
                             design_manifest_sha256: str, round_index: int) -> None:
    """Require all source shards to use the same input and code identity."""
    expected = {row['source_id'] for row in design_manifest['sources']}
    if len(summaries) != len(expected) or {row['source_id'] for row in summaries} != expected:
        raise ValueError('feedback GPU source summaries incomplete')
    commits = set()
    for row in summaries:
        source_id = row['source_id']
        roster = {f"{arm['seed_id']}_{arm['arm']}" for arm in arms['arms']
                  if arm['seed_id'].startswith(f'source_{source_id:04d}_')}
        runtime = row['runtime']
        if (row.get('arms_sha256') != arms_sha256 or row.get('manifest_sha256') != design_manifest_sha256
                or row.get('round') != round_index or not row.get('code_commit') or not row.get('cluster_id')
                or runtime.get('contract') != {'precision': 'fp32', 'amp_enabled': False, 'tf32_enabled': False}
                or runtime.get('matmul_tf32_enabled') is not False or runtime.get('cudnn_tf32_enabled') is not False
                or len(row['arms']) != len(roster) or {item['arm'] for item in row['arms']} != roster):
            raise ValueError('feedback GPU source summary identity mismatch')
        commits.add(row['code_commit'])
    if len(commits) != 1:
        raise ValueError('feedback GPU code identities differ')


def lock_feedback_round(arms: Mapping[str, Any], requests: Mapping[str, Mapping[str, Any]], *,
                        design_manifest: Mapping[str, Any], design_lock: Mapping[str, Any],
                        base_sha256: str, design_manifest_sha256: str, round_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind every ready arm's saved request to its prior observations and candidate."""
    if (arms.get('scope') != 'matched_feedback_arms_initialized'
            or arms.get('base_config_sha256') != base_sha256
            or design_manifest.get('base_config_sha256') != base_sha256
            or design_lock.get('manifest_sha256') != design_manifest_sha256):
        raise ValueError('feedback handoff input identity mismatch')
    source_ids = {f"source_{s['source_id']:04d}": s['source_id'] for s in design_manifest['sources']}
    synthetic, records, names = [], [], set()
    for arm in arms['arms']:
        name = f"{arm['seed_id']}_{arm['arm']}"
        source_name = arm['seed_id'].split('_initial_', 1)[0]
        if name in names or source_name not in source_ids:
            raise ValueError('feedback arm identity duplicated or source missing')
        names.add(name)
        synthetic.append({'trial_id': name, 'source_id': source_ids[source_name]})
        item = {'label': f'{name}/round_{round_index}', 'trial_id': name, 'role': f'round_{round_index}'}
        if arm['status'] != 'ready':
            records.append({**item, 'candidate': None, 'status': arm['status']})
            continue
        request = requests.get(name)
        if request is None:
            raise ValueError('ready feedback arm has no frozen request')
        if request.get('failure_status') == 'no_valid_candidate':
            records.append({**item, 'candidate': None, 'status': 'no_valid_candidate'})
            continue
        frozen, bank, proposal = request['frozen'], request['observations'], request['proposal']
        recomputed = json.loads(_serialized(freeze_round(restore_feedback(arm['state']), proposal, bank)))
        if (bank != arm['observations'] or len(arm['state']['history']) != round_index
                or frozen != recomputed):
            raise ValueError('frozen feedback request does not match arm state')
        prediction, candidate = frozen['prefit_prediction'], frozen['candidate']
        item.update(point=prediction['point'], candidate=candidate,
                    surrogate_edges={key: prediction[key] for key in ('hard_edges', 'soft_exact_edges', 'soft_smooth_edges')},
                    frozen_sha256=frozen['frozen_sha256'])
        if frozen['cached_observation'] is not None:
            item.update(candidate=None, status='cached_physical_observation')
        records.append(item)
    if set(requests) - names:
        raise ValueError('unexpected feedback request')
    manifest = {'base_config_sha256': base_sha256, 'trials': synthetic,
                'parent_design_manifest_sha256': design_manifest_sha256}
    manifest_sha = hashlib.sha256(_serialized(manifest)).hexdigest()
    lock = {'scope': 'locked_paired_surrogate_campaign', 'purpose': 'feedback_round',
            'manifest_sha256': manifest_sha, 'round_index': round_index,
            'k_points': design_lock['k_points'], 'records': records}
    return manifest, lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('arms', 'design-manifest', 'design-lock', 'base', 'request-directory', 'output-directory'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--round', type=int, required=True)
    args = parser.parse_args()
    raw_arms, raw_design, raw_lock, raw_base = (Path(getattr(args, key)).read_bytes()
        for key in ('arms', 'design_manifest', 'design_lock', 'base'))
    arms, design, design_lock, base = (json.loads(raw) for raw in (raw_arms, raw_design, raw_lock, raw_base))
    summaries = [json.loads(path.read_bytes()) for path in Path(args.request_directory).glob('source_*_summary.json')]
    validate_round_summaries(summaries, arms=arms, design_manifest=design,
        arms_sha256=hashlib.sha256(raw_arms).hexdigest(), design_manifest_sha256=hashlib.sha256(raw_design).hexdigest(),
        round_index=args.round)
    requests = {}
    for path in Path(args.request_directory).glob('*/request.json'):
        request = json.loads(path.read_bytes())
        request['proposal'] = json.loads((path.parent/'proposal.json').read_bytes())
        requests[path.parent.name] = request
    for path in Path(args.request_directory).glob('*/failure.json'):
        if path.parent.name in requests:
            raise ValueError('feedback arm has both a request and a failure')
        requests[path.parent.name] = {'failure_status': json.loads(path.read_bytes())['status']}
    design_sha, base_sha = hashlib.sha256(raw_design).hexdigest(), hashlib.sha256(raw_base).hexdigest()
    synthetic, lock = lock_feedback_round(arms, requests, design_manifest=design, design_lock=design_lock,
        base_sha256=base_sha, design_manifest_sha256=design_sha, round_index=args.round)
    synthetic_raw, lock_raw = _serialized(synthetic), _serialized(lock)
    physical, protocol = build_campaign_request(lock, synthetic, base,
        lock_sha256=hashlib.sha256(lock_raw).hexdigest(), manifest_sha256=hashlib.sha256(synthetic_raw).hexdigest(),
        base_sha256=base_sha)
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    for name, payload in (('round_manifest.json', synthetic), ('round_lock.json', lock),
                          ('manifest.json', physical), ('protocol.json', protocol)):
        (out/name).write_bytes(_serialized(payload))


if __name__ == '__main__':
    main()
