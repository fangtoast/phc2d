"""Integration of Fourier proposal, numerical evaluator and feedback state.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import numpy as np
import pytest
import json
import base64
import hashlib

torch = pytest.importorskip('torch')

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.feedback_evaluator import evaluate_feedback_candidate
from phc2d.inverse.feedback_loop import run_feedback_loop
from phc2d.inverse.feedback_state import start_feedback
from phc2d.inverse.feedback_batch_dispatch import FeedbackBatchState
from phc2d.inverse.feedback_handoff import build_robust_solver_manifest
from phc2d.inverse.feedback_journal import create_batch_journal, append_batch_event, verify_batch_journal
from phc2d.inverse.audit_perturb_refine import run_case as run_offset_case
from phc2d.inverse.resource_ledger import ResourceAmount, ResourceLedger
from phc2d.inverse.fourier_feedback_proposal import propose_fourier_candidate, run_budgeted_nominal_feedback_proposal
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


class ToyPredictor:
    def predict_tensor(self, raster):
        fill = raster.mean(dim=(1, 2))
        return torch.stack((.2+.1*fill, .8-.1*fill), dim=1).unsqueeze(1)


def test_actual_fourier_proposal_flows_through_solver_gate_to_accepted_incumbent():
    source = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    config = FullCellMpbBandSolverConfig(num_bands=2, resolution=16, polarization='TE',
        epsilon_air=source.epsilon_air, epsilon_material=source.epsilon_material, eigensolver_tolerance=1e-7)
    options = {'verification_resolution': 24, 'initial_spacing': .25, 'min_levels': 2,
               'max_levels': 2, 'required_stable_levels': 2, 'edge_tolerance': 1,
               'resolution_tolerance': 1e-6}
    points = np.array([[0, 0], [.25, 0]])
    def solver(item, *, k_points, config):
        fill = float(item.rasterize((16, 16)).mean())
        return np.tile([.2+.1*fill, .8-.1*fill], (len(k_points), 1)), {'geometry_hash': item.geometry_hash}
    def evaluate(item):
        return evaluate_feedback_candidate(item, k_points=points, config=config, lower_band=1,
                                           refinement=options, solver=solver)
    anchor = evaluate(source)
    assert anchor['status'] == 'numerically_stable'
    threshold = source.parameters['threshold']
    state = start_feedback(incumbent_hash=source.parameter_hash, incumbent_parameters=[threshold],
        incumbent_observation=anchor['observation'], target_frequency=.5, radius=.1,
        min_radius=.01, max_radius=.2, candidate_budget=1)
    candidates, raw_records, saved = {}, [], []
    def propose(current):
        result = propose_fourier_candidate(current, source, ToyPredictor(), coefficient_keys=(),
            global_bounds=((threshold-.5, threshold+.5),), origin=[0], scales=[1],
            max_mode_radius=1, seed=3, population_size=5, surrogate_budget=10,
            raster_method='master_block_mean_v1', source_resolution=16, input_resolution=8,
            material_fraction_bounds=(0, 1), lower_band=1, surrogate_contract_sha256='model')
        assert result['status'] == 'surrogate_proposal_only'
        candidates[result['proposal']['parameter_hash']] = FullCellStructureSpec.from_mapping(result['structure'])
        return result['proposal']
    def solver_attempt(pending):
        result = evaluate(candidates[pending.pending.parameter_hash])
        raw_records.append(result)
        return result['observation']
    final, status = run_feedback_loop(state, propose=propose, evaluate=solver_attempt, persist=saved.append)
    assert status == 'budget_exhausted'
    assert final.incumbent_hash != source.parameter_hash
    assert final.history[0]['accepted'] and raw_records[0]['status'] == 'numerically_stable'
    assert saved[0].pending is not None and saved[-1].pending is None


def test_two_budgeted_proposals_wait_for_solver_feedback(tmp_path):
    source = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    config = FullCellMpbBandSolverConfig(num_bands=2, resolution=16, polarization='TE',
        epsilon_air=source.epsilon_air, epsilon_material=source.epsilon_material, eigensolver_tolerance=1e-7)
    options = {'verification_resolution': 24, 'initial_spacing': .25, 'min_levels': 2,
               'max_levels': 2, 'required_stable_levels': 2, 'edge_tolerance': 1,
               'resolution_tolerance': 1e-6}
    points = np.array([[0, 0], [.25, 0]])
    def solver(item, *, k_points, config):
        fill = float(item.rasterize((16, 16)).mean())
        return np.tile([.2+.1*fill, .8-.1*fill], (len(k_points), 1)), {'geometry_hash': item.geometry_hash}
    def evaluate(item):
        return evaluate_feedback_candidate(item, k_points=points, config=config, lower_band=1,
                                           refinement=options, solver=solver)
    threshold = source.parameters['threshold']
    state = start_feedback(incumbent_hash=source.parameter_hash, incumbent_parameters=[threshold],
        incumbent_observation=evaluate(source)['observation'], target_frequency=.5, radius=.1,
        min_radius=.01, max_radius=.2, candidate_budget=2)
    candidates = {source.parameter_hash: source}
    events, pending_states = [], []
    def propose(current):
        round_index = len(current.history)
        events.append(('proposal', round_index, current.incumbent_hash))
        result = run_budgeted_nominal_feedback_proposal(current, candidates[current.incumbent_hash], ToyPredictor(),
            output_root=tmp_path/f'round_{round_index}', provenance={'scope': 'toy'}, coefficient_keys=(),
            global_bounds=((threshold-.5, threshold+.5),), origin=[0], scales=[1], max_mode_radius=1,
            seed=3+round_index, population_size=5, surrogate_budget=10, raster_method='master_block_mean_v1',
            source_resolution=16, input_resolution=8, material_fraction_bounds=(0, 1), lower_band=1,
            surrogate_contract_sha256='model', perturbation_offsets=(0.,))['result']
        if result['proposal'] is not None:
            candidate = FullCellStructureSpec.from_mapping(result['structure'])
            candidates[candidate.parameter_hash] = candidate
        return result['proposal']
    def solve(pending):
        events.append(('solve', len(pending.history), pending.pending.parameter_hash))
        return evaluate(candidates[pending.pending.parameter_hash])['observation']
    final, status = run_feedback_loop(state, propose=propose, evaluate=solve, persist=pending_states.append)
    assert len(final.history) == 2 and status == 'budget_exhausted'
    assert [item[0] for item in events] == ['proposal', 'solve', 'proposal', 'solve']
    for round_index in range(2):
        cost = json.loads((tmp_path/f'round_{round_index}'/'cost.json').read_bytes())
        assert cost['inferences'] <= 10 and cost['archive_error'] is None
        assert (tmp_path/f'round_{round_index}'/'predictions.npz').is_file()


def test_budgeted_proposal_through_persisted_batch_and_restart(tmp_path):
    source = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    config = FullCellMpbBandSolverConfig(num_bands=2, resolution=16, polarization='TE',
        epsilon_air=source.epsilon_air, epsilon_material=source.epsilon_material, eigensolver_tolerance=1e-7)
    refinement = {'verification_resolution': 24, 'initial_spacing': .25, 'min_levels': 2,
        'max_levels': 2, 'required_stable_levels': 2, 'edge_tolerance': 1, 'resolution_tolerance': 1e-6}
    points = np.array([[0., 0.], [.25, 0.]])
    def solver(grid, *, k_points, config):
        fill = float(grid.mean())
        meta = {'grid_sha256': hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest(),
            'resolution': config.resolution, 'polarization': config.polarization, 'num_bands': config.num_bands,
            'mesh_size': config.mesh_size, 'band_guard_count': config.band_guard_count,
            'band_order_policy': config.band_order_policy, 'eigensolver_tolerance': config.eigensolver_tolerance,
            'epsilon_air': config.epsilon_air, 'epsilon_material': config.epsilon_material,
            'solver_backend': 'mock', 'solver_version': '1'}
        return np.tile([.2+.1*fill, .8-.1*fill], (len(k_points), 1)), meta
    anchor = evaluate_feedback_candidate(source, k_points=points, config=config, lower_band=1,
        refinement=refinement, offsets=(0.,), raster_resolution=16, solver=solver)['observation']
    threshold = source.parameters['threshold']
    feedback = start_feedback(incumbent_hash=source.parameter_hash, incumbent_parameters=[threshold],
        incumbent_observation=anchor, target_frequency=.5, radius=.1, min_radius=.01,
        max_radius=.2, candidate_budget=2)
    batch = FeedbackBatchState({'lane': feedback}, {}, ResourceLedger(ResourceAmount(80, 0)), 1, 2)
    journal = tmp_path/'journal'
    initial = create_batch_journal(journal, initial=batch)
    head = initial
    def event(operation, payload):
        nonlocal head
        result, head = append_batch_event(journal, initial_sha256=initial, expected_head=head,
            batch_id='batch', operation=operation, payload=payload)
        return result
    proposal = run_budgeted_nominal_feedback_proposal(feedback, source, ToyPredictor(),
        output_root=tmp_path/'proposal', provenance={'scope': 'toy'}, coefficient_keys=(),
        global_bounds=((threshold-.5, threshold+.5),), origin=[0], scales=[1], max_mode_radius=1,
        seed=3, population_size=5, surrogate_budget=10, raster_method='master_block_mean_v1',
        source_resolution=16, input_resolution=8, material_fraction_bounds=(0, 1), lower_band=1,
        surrogate_contract_sha256='model', perturbation_offsets=(0.,))['result']
    candidate = FullCellStructureSpec.from_mapping(proposal['structure'])
    event('prepare', {'task_id': 'task', 'cluster_id': 'toy', 'workers': {'worker': 'lane'},
        'proposals': {'worker': proposal['proposal']}, 'reservation': {'cpu_core_seconds': 40, 'gpu_seconds': 0}})
    event('submission_unknown', {'reason': 'toy submit'})
    restored, _ = verify_batch_journal(journal, initial_sha256=initial, expected_head=head)
    assert restored.batches['batch'].phase == 'submission_unknown'
    pending = event('confirm', {'job_id': '100', 'confirmation_source': 'toy scheduler'}).lanes['lane']
    manifest = build_robust_solver_manifest(pending, candidate, k_points=points, config=config,
                                            lower_band=1, source_index=3)
    mp, pp = tmp_path/'manifest.json', tmp_path/'protocol.json'
    mp.write_text(json.dumps(manifest), encoding='utf8')
    pp.write_text(json.dumps({'offsets': [0.0], 'raster_resolution': 16, 'refinement': refinement}), encoding='utf8')
    mh, ph = (hashlib.sha256(p.read_bytes()).hexdigest() for p in (mp, pp))
    output = run_offset_case(mp, expected_sha256=mh, protocol_path=pp, expected_protocol_sha256=ph,
                             case_index=0, output_root=tmp_path/'cases', solver=solver)
    def encoded(raw):
        return base64.b64encode(raw).decode('ascii')
    event('reconcile', {'workers': {'worker': {'exit_code': 0, 'manifest': encoded(mp.read_bytes()),
        'manifest_sha256': mh, 'protocol': encoded(pp.read_bytes()), 'protocol_sha256': ph,
        'result': encoded((output/'results.json').read_bytes()), 'failure_reason': None}}})
    settled = event('settle', {'accounting': encoded(b'100|COMPLETED|10|4|cpu=4,node=1\n'), 'expected_gpus': 0})
    restored, count = verify_batch_journal(journal, initial_sha256=initial, expected_head=head)
    assert restored == settled and count == 5
    assert restored.lanes['lane'].remaining_evaluations == 1
    assert restored.ledger.totals()['actual'] == ResourceAmount(40, 0)
