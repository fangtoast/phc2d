"""Test feedback candidates are locked before solver handoff and caches are skipped."""
import copy
import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.differentiable_campaign_handoff import build_campaign_request
from phc2d.inverse.differentiable_feedback_handoff import _serialized, lock_feedback_round
from phc2d.inverse.differentiable_feedback_round import freeze_round
from phc2d.inverse.feedback_state import start_feedback


def test_frozen_feedback_handoff_uses_unique_physical_cases():
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    grid = np.ascontiguousarray(source.rasterize((64, 64)), dtype=np.uint8)
    candidate = {'status': 'surrogate_only', 'index': 19, 'structure': source.to_mapping(),
                 'parameter_hash': source.parameter_hash, 'master_grid_sha256': hashlib.sha256(grid.tobytes()).hexdigest()}
    physical = {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': .4,
                'refined_upper_edge_frequency': .6, 'last_refinement_stable': True,
                'resolution_stable': True, 'adaptive_refinement_converged': True},
                'provenance': {'parameter_hash': 'a', 'comparison_contract_sha256': 'physical'}}
    state = start_feedback(incumbent_hash='a', incumbent_parameters=[0, 0], incumbent_observation=physical,
        target_frequency=.5, radius=.2, min_radius=.025, max_radius=.2, candidate_budget=3,
        acceptance_policy='strict_physical_improvement')
    bank = [{'parameter_hash': 'a'}]
    digest = hashlib.sha256(json.dumps(bank, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    prediction = {'point': [.1, 0], 'hard_edges': [.38, .62], 'soft_exact_edges': [.37, .63],
                  'soft_smooth_edges': [.36, .64], 'corrected_smooth_edges': [.35, .65]}
    proposal = {'status': 'proposal_ready', 'observation_sha256': digest, 'selected_index': 19,
                'history': [{'parameters': [0, 0], 'objective': .1}] * 19 + [{'parameters': [.1, 0], 'objective': .2}],
                'hard_screening': {'candidates': [candidate] * 20}, 'prefit_prediction': prediction,
                'correction': {'center': [0, 0], 'radius': .2, 'surrogate_contract_sha256': 'model',
                               'comparison_contract_sha256': 'physical'}}
    frozen = json.loads(_serialized(freeze_round(state, proposal, bank)))
    name = 'source_0008_initial_0_adam_adam_residual'
    arms = {'scope': 'matched_feedback_arms_initialized', 'base_config_sha256': 'base', 'arms': [
        {'seed_id': 'source_0008_initial_0_adam', 'arm': 'adam_residual', 'status': 'ready',
         'state': asdict(state), 'observations': bank}]}
    requests = {name: {'frozen': frozen, 'observations': bank, 'proposal': proposal}}
    points = [[0., 0.]]
    design = {'base_config_sha256': 'base', 'sources': [{'source_id': 8}]}
    design_lock = {'manifest_sha256': 'design', 'k_points': points}
    synthetic, lock = lock_feedback_round(arms, requests, design_manifest=design, design_lock=design_lock,
                                          base_sha256='base', design_manifest_sha256='design', round_index=0)
    base = {'source_resolution': 64, 'k_points_sha256': hashlib.sha256(np.asarray(points, dtype=np.float32).tobytes()).hexdigest(),
            'physical_protocol': {'offsets': [0.0], 'raster_resolution': 64},
            'lower_band': 1, 'target_frequency': .5, 'solver_config': {}}
    manifest_sha = hashlib.sha256(_serialized(synthetic)).hexdigest()
    physical_request, _ = build_campaign_request(lock, synthetic, base, lock_sha256='lock',
        manifest_sha256=manifest_sha, base_sha256='base')
    assert len(physical_request['rows']) == 1
    bad = copy.deepcopy(requests); bad[name]['frozen']['prefit_prediction']['point'][0] += .01
    with pytest.raises(ValueError, match='does not match'):
        lock_feedback_round(arms, bad, design_manifest=design, design_lock=design_lock,
                            base_sha256='base', design_manifest_sha256='design', round_index=0)
    cached_bank = bank + [{'parameter_hash': candidate['parameter_hash'], 'parameters': [.1, 0]}]
    cached_proposal = copy.deepcopy(proposal)
    cached_proposal['observation_sha256'] = hashlib.sha256(json.dumps(
        cached_bank, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    cached = json.loads(_serialized(freeze_round(state, cached_proposal, cached_bank)))
    cached_arms = copy.deepcopy(arms); cached_arms['arms'][0]['observations'] = cached_bank
    _, cached_lock = lock_feedback_round(cached_arms, {name: {'frozen': cached,
        'observations': cached_bank, 'proposal': cached_proposal}}, design_manifest=design,
        design_lock=design_lock, base_sha256='base', design_manifest_sha256='design', round_index=0)
    assert cached_lock['records'][0]['candidate'] is None
    assert cached_lock['records'][0]['status'] == 'cached_physical_observation'
    _, failed_lock = lock_feedback_round(arms, {name: {'failure_status': 'no_valid_candidate'}},
        design_manifest=design, design_lock=design_lock, base_sha256='base',
        design_manifest_sha256='design', round_index=0)
    assert failed_lock['records'][0]['status'] == 'no_valid_candidate'
