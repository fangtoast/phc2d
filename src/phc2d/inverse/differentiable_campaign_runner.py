"""Run one source/method shard of a locked paired inverse campaign.

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
from typing import Any, Mapping

from phc2d.inverse.differentiable_mvp import load_design_inputs, run_shard
from phc2d.inverse.differentiable_robust import run_robust_shard


def run_trials(manifest: Mapping[str, Any], *, manifest_sha256: str, source_id: int,
               method: str, source: Any, predictor: Any, runtime: Mapping[str, Any],
               code_commit: str, cluster_id: str, output: Path) -> list[str]:
    """Save individual trials exclusively; resume only matching completed identities."""
    robust = manifest.get('scope') == 'preregistered_robust_design'
    methods = ('nominal', 'mean', 'worst') if robust else ('adam', 'de', 'random')
    if not manifest_sha256 or not code_commit or not cluster_id or method not in methods:
        raise ValueError('missing campaign identity or invalid method')
    trials = [row for row in manifest['trials'] if row['source_id'] == source_id and row['method'] == method]
    if len(trials) != 3 or {row['initial_index'] for row in trials} != {0, 1, 2}:
        raise ValueError('source/method trial roster mismatch')
    output.mkdir(parents=True, exist_ok=True)
    completed = []
    for row in trials:
        trial_id = row['trial_id']
        path = output / f'{trial_id}.json'
        failure = output / f'{trial_id}.failure.json'
        identity = {'trial_id': trial_id, 'manifest_sha256': manifest_sha256,
                    'code_commit': code_commit, 'cluster_id': cluster_id}
        if failure.exists():
            raise FileExistsError(f'prior trial failure must be reconciled: {trial_id}')
        if path.exists():
            previous = json.loads(path.read_bytes())
            if any(previous.get(key) != value for key, value in identity.items()):
                raise ValueError(f'existing trial identity mismatch: {trial_id}')
            completed.append(trial_id)
            continue
        if row['status'] != 'ready':
            payload = {**identity, 'status': row['status'], 'initial_index': row['initial_index']}
        else:
            cfg = row['config']
            if (cfg['initial'] is None or cfg['evaluations'] != (manifest['evaluations'] if robust else 60)
                    or (robust and (cfg['offsets'] != manifest['offsets'] or row['mode'] != method))
                    or cfg['checkpoint_sha256'] != predictor.checkpoint_sha256):
                raise ValueError('invalid trial config or checkpoint identity')
            try:
                result = (run_robust_shard(cfg, source, predictor, mode=method) if robust
                          else run_shard(cfg, source, predictor, role=method))
                if len(result['history']) != cfg['evaluations']:
                    raise ValueError('incomplete optimization trajectory')
                result.update(identity, status='completed', source_id=source_id,
                              source_parameter_hash=source.parameter_hash, initial_index=row['initial_index'],
                              source_sha256=cfg['source_sha256'], checkpoint_sha256=predictor.checkpoint_sha256,
                              k_points_sha256=cfg['k_points_sha256'],
                              k_points=predictor.k_points.detach().cpu().tolist(),
                              config=cfg, runtime=dict(runtime))
                payload = result
            except Exception as exc:
                with failure.open('x', encoding='utf8') as stream:
                    json.dump({**identity, 'error_type': type(exc).__name__, 'message': str(exc)}, stream)
                raise
        # Exclusive writes make interrupted outputs visible and prevent silent replacement.
        with path.open('x', encoding='utf8') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
        completed.append(trial_id)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'source', 'checkpoint', 'method', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--source-id', type=int, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    raw = Path(args.manifest).read_bytes()
    manifest = json.loads(raw)
    matches = [row for row in manifest['trials'] if row['source_id'] == args.source_id
               and row['method'] == args.method and row['status'] == 'ready']
    if not matches:
        raise ValueError('no ready trial for selected source and method')
    source, predictor, runtime = load_design_inputs(matches[0]['config'], args.source,
                                                     args.checkpoint, device=args.device)
    source_rows = [row for row in manifest['sources'] if row['source_id'] == args.source_id]
    if len(source_rows) != 1 or source.parameter_hash != source_rows[0]['source_parameter_hash']:
        raise ValueError('source identity differs from campaign')
    result = run_trials(manifest, manifest_sha256=hashlib.sha256(raw).hexdigest(), source_id=args.source_id,
                        method=args.method, source=source, predictor=predictor, runtime=runtime,
                        code_commit=os.environ['PHC2D_DEPLOYED_GIT_COMMIT'],
                        cluster_id=os.environ['PHC2D_CLUSTER_ID'], output=Path(args.output))
    print(json.dumps({'completed_trial_ids': result}))


if __name__ == '__main__':
    main()
