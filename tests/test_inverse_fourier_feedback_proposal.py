"""End-to-end proposal identity and trust bounds for Fourier feedback.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest
import numpy as np

torch = pytest.importorskip('torch')

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.feedback_state import propose_feedback, start_feedback
from phc2d.inverse.fourier_feedback_proposal import propose_fourier_candidate
from phc2d.inverse.local_edge_correction import LocalEdgeCorrection


class ToyPredictor:
    def predict_tensor(self, raster):
        mean = raster.mean(dim=(1, 2))
        return torch.stack((.2 + .1*mean, .8 - .1*mean), dim=1).unsqueeze(1)


def test_search_proposal_respects_frozen_coordinates_and_feedback_identity():
    source = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    threshold = source.parameters['threshold']
    observation = {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': .3,
                           'refined_upper_edge_frequency': .7, 'last_refinement_stable': True,
                           'resolution_stable': True, 'adaptive_refinement_converged': True},
                   'provenance': {'parameter_hash': source.parameter_hash,
                                  'comparison_contract_sha256': 'solver'}}
    state = start_feedback(incumbent_hash=source.parameter_hash, incumbent_parameters=[threshold],
        incumbent_observation=observation, target_frequency=.5, radius=.1, min_radius=.01,
        max_radius=.2, candidate_budget=1)
    result = propose_fourier_candidate(state, source, ToyPredictor(), coefficient_keys=(),
        global_bounds=((threshold-.5, threshold+.5),), origin=[0], scales=[1],
        max_mode_radius=1, seed=3, population_size=5, surrogate_budget=10,
        raster_method='master_block_mean_v1', source_resolution=16, input_resolution=8,
        material_fraction_bounds=(0, 1), lower_band=1, surrogate_contract_sha256='model')
    assert result['status'] == 'surrogate_proposal_only'
    candidate = FullCellStructureSpec.from_mapping(result['structure'])
    assert candidate.parameter_hash == result['proposal']['parameter_hash']
    assert abs(result['proposal']['parameters'][0] - threshold) <= .1
    assert len(result['trace']) <= 10
    pending = propose_feedback(state, **result['proposal'])
    assert pending.pending.parameter_hash == candidate.parameter_hash
    correction = LocalEdgeCorrection(np.array([threshold]), .1, np.array([[-.02, .02], [0., 0.]]),
                                     'solver', 'model', 1)
    corrected = propose_fourier_candidate(state, source, ToyPredictor(), coefficient_keys=(),
        global_bounds=((threshold-.5, threshold+.5),), origin=[0], scales=[1],
        max_mode_radius=1, seed=3, population_size=5, surrogate_budget=10,
        raster_method='master_block_mean_v1', source_resolution=16, input_resolution=8,
        material_fraction_bounds=(0, 1), lower_band=1, surrogate_contract_sha256='model',
        edge_correction=correction)
    assert corrected['edge_correction_applied']
    assert corrected['proposal']['predicted_center'] == pytest.approx(result['proposal']['predicted_center'] + .08)
    invalid = LocalEdgeCorrection(np.array([threshold]), .1, np.zeros((2, 2)), 'wrong', 'model', 1)
    with pytest.raises(ValueError, match='correction contract'):
        propose_fourier_candidate(state, source, ToyPredictor(), coefficient_keys=(),
            global_bounds=((threshold-.5, threshold+.5),), origin=[0], scales=[1],
            max_mode_radius=1, seed=3, population_size=5, surrogate_budget=10,
            raster_method='master_block_mean_v1', source_resolution=16, input_resolution=8,
            material_fraction_bounds=(0, 1), lower_band=1, surrogate_contract_sha256='model',
            edge_correction=invalid)


def test_inconsistent_normalized_source_coordinates_are_rejected():
    source = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    observation = {'row': {'lower_band_one_based': 1, 'refined_lower_edge_frequency': .3,
                           'refined_upper_edge_frequency': .7, 'last_refinement_stable': True,
                           'resolution_stable': True, 'adaptive_refinement_converged': True},
                   'provenance': {'parameter_hash': source.parameter_hash,
                                  'comparison_contract_sha256': 'solver'}}
    state = start_feedback(incumbent_hash=source.parameter_hash, incumbent_parameters=[0],
        incumbent_observation=observation, target_frequency=.5, radius=.1, min_radius=.01,
        max_radius=.2, candidate_budget=1)
    with pytest.raises(ValueError, match='coordinates'):
        propose_fourier_candidate(state, source, ToyPredictor(), coefficient_keys=(),
            global_bounds=((-1, 1),), origin=[0], scales=[1], max_mode_radius=1,
            seed=1, population_size=5, surrogate_budget=5, raster_method='master_block_mean_v1',
            source_resolution=16, input_resolution=8, material_fraction_bounds=(0, 1),
            lower_band=1, surrogate_contract_sha256='model')
