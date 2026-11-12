"""Run one source's frozen feedback proposals on a loaded surrogate.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from phc2d.inverse.differentiable_feedback_campaign import run_feedback_arm_round
from phc2d.inverse.differentiable_mvp import load_design_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('arms', 'base', 'campaign', 'design-manifest', 'source', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--source-id', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    raw_arms, raw_base, raw_campaign, raw_manifest = (Path(getattr(args, name)).read_bytes()
        for name in ('arms', 'base', 'campaign', 'design_manifest'))
    payload, base, campaign, manifest = (json.loads(raw) for raw in
        (raw_arms, raw_base, raw_campaign, raw_manifest))
    if (payload.get('scope') != 'matched_feedback_arms_initialized'
            or payload.get('base_config_sha256') != hashlib.sha256(raw_base).hexdigest()
            or payload.get('campaign_sha256') != hashlib.sha256(raw_campaign).hexdigest()
            or manifest.get('base_config_sha256') != hashlib.sha256(raw_base).hexdigest()):
        raise ValueError('feedback campaign input identity mismatch')
    sources = [row for row in manifest['sources'] if row['source_id'] == args.source_id]
    trials = [row for row in manifest['trials'] if row['source_id'] == args.source_id and row['status'] == 'ready']
    if len(sources) != 1 or not trials:
        raise ValueError('feedback source is absent or invalid')
    source, predictor, runtime = load_design_inputs(trials[0]['config'], args.source, args.checkpoint, device=args.device)
    if source.parameter_hash != sources[0]['source_parameter_hash']:
        raise ValueError('feedback source parameter identity mismatch')
    selected = [arm for arm in payload['arms'] if arm['seed_id'].startswith(f'source_{args.source_id:04d}_')]
    if len(selected) != len(sources[0]['initials']) * len(campaign['arms']):
        raise ValueError('feedback arm roster is incomplete')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    identity = {'arms_sha256': hashlib.sha256(raw_arms).hexdigest(),
                'manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(),
                'code_commit': os.environ['PHC2D_DEPLOYED_GIT_COMMIT'],
                'cluster_id': os.environ['PHC2D_CLUSTER_ID'], 'runtime': runtime,
                'source_id': args.source_id, 'round': args.round}
    summary = []
    for arm in selected:
        name = f"{arm['seed_id']}_{arm['arm']}"
        if arm['status'] != 'ready':
            summary.append({'arm': name, 'status': arm['status']})
            continue
        result = run_feedback_arm_round(arm, base=base, campaign=campaign, source_meta=sources[0],
            source=source, predictor=predictor, round_index=args.round, output=root/name)
        summary.append({'arm': name, **result})
    destination = root/f'source_{args.source_id:04d}_summary.json'
    if destination.exists():
        prior = json.loads(destination.read_bytes())
        if (any(prior.get(key) != value for key, value in identity.items())
                or [item['arm'] for item in prior['arms']] != [item['arm'] for item in summary]):
            raise ValueError('existing source summary identity mismatch')
        return
    with destination.open('x', encoding='utf8') as stream:
        json.dump({**identity, 'arms': summary}, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
