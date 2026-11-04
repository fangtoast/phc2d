"""Rebind authenticated MPB results to a frozen feedback seed manifest.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phc2d.inverse.differentiable_feedback_reuse import rebind_seed_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('target-manifest', 'reference-manifest', 'fresh-manifest', 'reuse-plan',
                 'reference-results', 'fresh-results', 'output-directory'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    target_raw, reference_raw, fresh_raw, plan_raw = (Path(getattr(args, name)).read_bytes()
        for name in ('target_manifest', 'reference_manifest', 'fresh_manifest', 'reuse_plan'))
    def load(root: str) -> dict:
        found = {}
        for path in Path(root).glob('case_*/results.json'):
            raw = path.read_bytes()
            item = json.loads(raw)
            if item['case_index'] in found:
                raise ValueError('duplicate physical result')
            found[item['case_index']] = (item, hashlib.sha256(raw).hexdigest())
        return found
    derived = rebind_seed_results(json.loads(target_raw), json.loads(reference_raw), json.loads(fresh_raw),
        json.loads(plan_raw), target_sha256=hashlib.sha256(target_raw).hexdigest(),
        reference_sha256=hashlib.sha256(reference_raw).hexdigest(), fresh_sha256=hashlib.sha256(fresh_raw).hexdigest(),
        reference_results=load(args.reference_results), fresh_results=load(args.fresh_results))
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    for index, payload in derived.items():
        case = output/f'case_{index:04d}'
        case.mkdir()
        (case/'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'derived_cases': len(derived), 'target_cases': len(json.loads(target_raw)['rows'])}))


if __name__ == '__main__':
    main()
