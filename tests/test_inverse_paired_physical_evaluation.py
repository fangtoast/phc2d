"""Exercise locked paired statistics using explicit mock solver records.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from phc2d.inverse.audit_perturb_refine import run_case
from phc2d.inverse.paired_physical_handoff import build_paired_physical_batch
from phc2d.inverse.paired_physical_evaluation import evaluate_locked_paired_physics
from phc2d.inverse.robust_cross_environment import compare_locked_robust_environments


def test_locked_comparison_requires_every_physical_gate(tmp_path):
    fixture = runpy.run_path(str(Path(__file__).with_name('test_inverse_paired_physical_handoff.py')))['fixture_inputs']
    sources, locks, kwargs = fixture()
    sources = sources[:2]
    for source in sources:
        source['stratum'] = 'positive'
    locks = [row for row in locks if row['source_index'] in (0, 1)]
    same = next(row for row in locks if row['source_index'] == 0 and row['model_seed'] == 1)
    same['random_choice'] = dict(same['surrogate_choice'])
    kwargs['expected_source_counts'] = {'positive':2}
    batch = build_paired_physical_batch(sources, locks, **kwargs)
    batch_raw = json.dumps(batch, sort_keys=True, separators=(',', ':')).encode()
    manifest_raw = json.dumps(batch['manifest'], sort_keys=True, separators=(',', ':')).encode()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    manifest_path, protocol_path = tmp_path/'manifest.json', tmp_path/'protocol.json'
    manifest_path.write_bytes(manifest_raw)
    protocol_path.write_bytes(kwargs['protocol_bytes'])

    def solver(grid, *, k_points, config, fail=False, shift=0.):
        fill = float(grid.mean())
        change = .01 if fail and config.resolution == 48 else 0.
        metadata = {'grid_sha256':hashlib.sha256(np.ascontiguousarray(grid,dtype=np.uint8).tobytes()).hexdigest(),
                    'resolution':config.resolution,'polarization':config.polarization,'num_bands':config.num_bands,
                    'mesh_size':config.mesh_size,'band_guard_count':config.band_guard_count,
                    'band_order_policy':config.band_order_policy,'eigensolver_tolerance':config.eigensolver_tolerance,
                    'epsilon_air':config.epsilon_air,'epsilon_material':config.epsilon_material,
                    'solver_backend':'mock','solver_version':'1'}
        return np.tile([.2+.1*fill+change+shift,.8-.1*fill+shift],(len(k_points),1)), metadata

    results = {}
    for index in range(len(batch['manifest']['rows'])):
        path = run_case(manifest_path,expected_sha256=manifest_sha,protocol_path=protocol_path,
            expected_protocol_sha256=kwargs['protocol_sha256'],case_index=index,output_root=tmp_path/'ok',solver=solver)
        results[index] = (path/'results.json').read_bytes()
    args = {'batch_bytes':batch_raw,'batch_sha256':hashlib.sha256(batch_raw).hexdigest(),
        'manifest_bytes':manifest_raw,'manifest_sha256':manifest_sha,'protocol_bytes':kwargs['protocol_bytes'],
        'protocol_sha256':kwargs['protocol_sha256'],'results_by_case':results,
        'source_strata':{0:'positive',1:'positive'},'seeds':[0,1],
        'expected_counts':{'positive':2},'stratum_weights':{'positive':1.},
        'bootstrap_seed':99,'bootstrap_samples':100,'expected_solver_identity':{'solver_backend':'mock','solver_version':'1'}}
    actual = evaluate_locked_paired_physics(**args)
    assert actual['status'] == 'paired_statistics_available' and actual['p2_gate'] == 'not_evaluated'
    assert actual['statistics']['source_seed_count'] == 4
    differences = [r['selected_gain']-r['random_expected_gain'] for r in actual['groups']]
    assert actual['statistics']['selected_minus_random_mean'] == pytest.approx(np.mean(differences))
    same_group = next(row for row in actual['groups'] if row['source_index'] == 0 and row['seed'] == 1)
    assert same_group['selected_gain'] == same_group['random_expected_gain']
    cross_ids = [r['case_index'] for r in batch['cross_environment_roles']]
    cross_args = {'batch_bytes':batch_raw,'batch_sha256':args['batch_sha256'],
        'manifest_bytes':manifest_raw,'manifest_sha256':manifest_sha,'protocol_bytes':kwargs['protocol_bytes'],
        'protocol_sha256':kwargs['protocol_sha256'], 'reference_results':{i:results[i] for i in cross_ids},
        'replay_results':{i:results[i] for i in cross_ids},'replay_exit_codes':{i:0 for i in cross_ids},
        'expected_case_count':len(cross_ids),'expected_solver_identity':args['expected_solver_identity'],
        'frequency_tolerance':1e-6}
    comparison = compare_locked_robust_environments(**cross_args)
    assert comparison['all_passed'] and comparison['p2_gate'] == 'not_evaluated'
    failed_case = cross_ids[0]
    exits = dict(cross_args['replay_exit_codes']); exits[failed_case] = 124
    replays = {i:raw for i,raw in cross_args['replay_results'].items() if i != failed_case}
    comparison = compare_locked_robust_environments(**dict(cross_args,replay_exit_codes=exits,replay_results=replays))
    assert not comparison['all_passed'] and comparison['cases'][0]['reason'] == 'replay_process_failed'
    with pytest.raises(ValueError,match='case or exit roster'):
        compare_locked_robust_environments(**dict(cross_args,replay_results={}))
    shifted = run_case(manifest_path,expected_sha256=manifest_sha,protocol_path=protocol_path,
        expected_protocol_sha256=kwargs['protocol_sha256'],case_index=failed_case,output_root=tmp_path/'shifted',
        solver=lambda grid,**kw:solver(grid,shift=.0001,**kw))
    replays = dict(cross_args['replay_results'])
    replays[failed_case] = (shifted/'results.json').read_bytes()
    comparison = compare_locked_robust_environments(**dict(cross_args,replay_results=replays))
    assert not comparison['all_passed'] and comparison['cases'][0]['offsets'][0]['max_frequency_difference'] > 1e-6
    with pytest.raises(ValueError,match='complete solver case roster'):
        evaluate_locked_paired_physics(**dict(args,results_by_case={0:results[0]}))
    tampered = dict(results)
    invalid = json.loads(tampered[0])
    invalid['parameter_hash'] = 'wrong'
    tampered[0] = json.dumps(invalid).encode()
    with pytest.raises(ValueError,match='solver result identity mismatch'):
        evaluate_locked_paired_physics(**dict(args,results_by_case=tampered))
    bad = run_case(manifest_path,expected_sha256=manifest_sha,protocol_path=protocol_path,
        expected_protocol_sha256=kwargs['protocol_sha256'],case_index=0,output_root=tmp_path/'bad',
        solver=lambda grid,**kw:solver(grid,fail=True,**kw))
    failed = dict(results, **{})
    failed[0] = (bad/'results.json').read_bytes()
    actual = evaluate_locked_paired_physics(**dict(args,results_by_case=failed))
    assert actual['status'] == 'unscoreable_numerical_gate_failures' and actual['failed_case_indices'] == [0]
    assert actual['statistics'] is None and actual['p2_gate'] == 'not_evaluated'
