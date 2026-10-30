"""Contracts for physically driven low-dimensional Fourier search.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest
import numpy as np

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.direct_fourier_search import search_fourier_physically
from phc2d.inverse.feedback_loop import SolverAttemptFailed
from phc2d.inverse.feedback_evaluator import evaluate_feedback_candidate
from phc2d.inverse.resource_ledger import (ResourceAmount, ResourceLedger, audit_slurm_ledger,
                                           reserve_resources, settle_slurm_resources)
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def source_and_config():
    source = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    threshold = source.parameters['threshold']
    return source, dict(coefficient_keys=(), bounds=((threshold - .2, threshold + .2),),
        max_mode_radius=1, seed=42, population_size=5, proposal_budget=10,
        solver_budget=10, raster_method='master_block_mean_v1', source_resolution=16,
        input_resolution=8, material_fraction_bounds=(0, 1), lower_band=1,
        target_frequency=.5, comparison_contract_sha256='physical-contract')


def observed(candidate, *, stable=True, contract='physical-contract'):
    fill = float(candidate.rasterize((16, 16)).mean())
    return {'status': 'numerically_stable' if stable else 'numerical_gate_failed',
            'observation': {'row': {'lower_band_one_based': 1,
                'refined_lower_edge_frequency': .2 + .1 * fill,
                'refined_upper_edge_frequency': .8 - .1 * fill,
                'last_refinement_stable': True, 'resolution_stable': stable,
                'adaptive_refinement_converged': True},
                'provenance': {'parameter_hash': candidate.parameter_hash,
                               'comparison_contract_sha256': contract}}}


def test_direct_search_uses_only_gated_physical_results_and_records_attempt_order():
    source, config = source_and_config()
    events = []
    def before(event):
        events.append(('pending', event['parameter_hash']))
    def evaluate(candidate):
        assert events[-1] == ('pending', candidate.parameter_hash)
        return observed(candidate)
    def after(event, outcome):
        events.append((outcome['status'], event['parameter_hash']))
    result = search_fourier_physically(source, evaluate, before_solve=before, after_solve=after, **config)
    assert result.status == 'search_complete'
    assert result.best_structure is not None and result.solver_attempts <= config['solver_budget']
    assert result.best_margin == max(row['physical_margin'] for row in result.evaluations
                                     if row['status'] == 'numerically_stable')
    assert len(events) == 2 * result.solver_attempts
    assert all(events[i][0] == 'pending' and events[i + 1][0] == 'numerically_stable'
               for i in range(0, len(events), 2))


def test_unstable_physics_never_becomes_best_and_contract_mismatch_fails_closed():
    source, config = source_and_config()
    result = search_fourier_physically(source, lambda c: observed(c, stable=False),
        before_solve=lambda _event: None, after_solve=lambda _event, _outcome: None, **config)
    assert result.status == 'no_numerically_valid_candidates' and result.best_structure is None
    with pytest.raises(ValueError, match='contract or band identity'):
        search_fourier_physically(source, lambda c: observed(c, contract='wrong'),
            before_solve=lambda _event: None, after_solve=lambda _event, _outcome: None, **config)


def test_failed_attempts_consume_budget_and_are_not_automatically_retried():
    source, config = source_and_config()
    config['solver_budget'] = 2
    events = []
    def fail(_candidate):
        raise SolverAttemptFailed('solver terminated')
    result = search_fourier_physically(source, fail, before_solve=lambda e: events.append(('pending', e)),
        after_solve=lambda e, o: events.append((o['status'], e)), **config)
    assert result.status == 'solver_budget_exhausted' and result.solver_attempts == 2
    assert [event[0] for event in events] == ['pending', 'solver_failure', 'pending', 'solver_failure']
    assert result.best_structure is None


def test_unexpected_solver_failure_preserves_pending_attempt_and_propagates():
    source, config = source_and_config()
    events = []
    def crash(_candidate):
        raise ValueError('solver artifact is malformed')
    with pytest.raises(ValueError, match='solver artifact is malformed'):
        search_fourier_physically(source, crash, before_solve=lambda e: events.append(('pending', e)),
            after_solve=lambda e, o: events.append((o['status'], e)), **config)
    assert len(events) == 1 and events[0][0] == 'pending'


def test_direct_search_integrates_with_refined_physical_evaluator():
    source, config = source_and_config()
    solver_config = FullCellMpbBandSolverConfig(num_bands=2, resolution=16, polarization='TE',
        epsilon_air=source.epsilon_air, epsilon_material=source.epsilon_material, eigensolver_tolerance=1e-7)
    refinement = {'verification_resolution': 24, 'initial_spacing': .25, 'min_levels': 2,
                  'max_levels': 2, 'required_stable_levels': 2, 'edge_tolerance': 1,
                  'resolution_tolerance': 1e-6}
    points = np.array([[0, 0], [.25, 0]])
    def solver(item, *, k_points, config):
        fill = float(item.rasterize((16, 16)).mean())
        return np.tile([.2 + .1 * fill, .8 - .1 * fill], (len(k_points), 1)), {'geometry_hash': item.geometry_hash}
    def evaluator(item):
        return evaluate_feedback_candidate(item, k_points=points, config=solver_config,
                                           lower_band=1, refinement=refinement, solver=solver)
    anchor = evaluator(source)
    config['comparison_contract_sha256'] = anchor['observation']['provenance']['comparison_contract_sha256']
    result = search_fourier_physically(source, evaluator, before_solve=lambda _event: None,
        after_solve=lambda _event, _outcome: None, **config)
    assert result.status == 'search_complete' and result.best_structure is not None
    assert all(row['status'] in {'numerically_stable', 'duplicate_cached'} for row in result.evaluations)


def test_direct_search_hooks_can_reserve_and_settle_each_solver_allocation():
    source, config = source_and_config()
    ledger = ResourceLedger(ResourceAmount(300, 0))
    accounting = {}
    def before(event):
        nonlocal ledger
        task = f"candidate-{event['candidate_index']}"
        ledger = reserve_resources(ledger, task_id=task, amount=ResourceAmount(30, 0))
    def after(event, outcome):
        nonlocal ledger
        assert outcome['status'] == 'numerically_stable'
        task = f"candidate-{event['candidate_index']}"
        job = f"700_{event['candidate_index']}"
        raw = f'{job}|COMPLETED|2|3|cpu=3,mem=8G\n'.encode()
        ledger = settle_slurm_resources(ledger, task_id=task, cluster_id='m9',
                                         accounting_bytes=raw, expected_gpus={job: 0})
        accounting[task] = {'cluster_id': 'm9', 'raw': raw, 'expected_gpus': {job: 0}}
    result = search_fourier_physically(source, observed, before_solve=before, after_solve=after, **config)
    audited = audit_slurm_ledger(ledger, accounting_by_task=accounting, require_settled=True)
    assert audited['settled_tasks'] == result.solver_attempts
    assert ledger.totals()['actual'].cpu_core_seconds == 6 * result.solver_attempts
