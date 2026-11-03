"""End-to-end proposal identity and trust bounds for Fourier feedback.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest
import numpy as np
import hashlib
import json

torch = pytest.importorskip('torch')

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.feedback_state import finish_feedback, propose_feedback, start_feedback
from phc2d.inverse.fourier_feedback_proposal import propose_fourier_candidate, run_budgeted_nominal_feedback_proposal
from phc2d.inverse.frozen_band_adapter import (SingleSampleBudgetedPredictor, audit_completed_inference_cost,
                                              audit_prediction_archive_against_calls)
from phc2d.inverse.surrogate_design import audit_nominal_prediction_archive
from phc2d.inverse.local_edge_correction import LocalEdgeCorrection
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids


class ToyPredictor:
    def predict_tensor(self, raster):
        mean = raster.mean(dim=(1, 2))
        return torch.stack((.2 + .1*mean, .8 - .1*mean), dim=1).unsqueeze(1)


def test_search_proposal_respects_frozen_coordinates_and_feedback_identity(tmp_path):
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
        max_radius=.2, candidate_budget=2)
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
    budgeted = SingleSampleBudgetedPredictor(ToyPredictor(), max_inferences=10)
    reused = propose_fourier_candidate(state, source, budgeted, coefficient_keys=(),
        global_bounds=((threshold-.5, threshold+.5),), origin=[0], scales=[1],
        max_mode_radius=1, seed=3, population_size=5, surrogate_budget=10,
        raster_method='master_block_mean_v1', source_resolution=16, input_resolution=8,
        material_fraction_bounds=(0, 1), lower_band=1, surrogate_contract_sha256='model',
        perturbation_offsets=(0.,))
    assert len(reused['trace']) == 10 and budgeted.inferences <= 10
    uncached = propose_fourier_candidate(state, source, ToyPredictor(), coefficient_keys=(),
        global_bounds=((threshold-.5, threshold+.5),), origin=[0], scales=[1],
        max_mode_radius=1, seed=3, population_size=5, surrogate_budget=10,
        raster_method='master_block_mean_v1', source_resolution=16, input_resolution=8,
        material_fraction_bounds=(0, 1), lower_band=1, surrogate_contract_sha256='model',
        perturbation_offsets=(0.,))
    assert reused == uncached
    kwargs = dict(coefficient_keys=(), global_bounds=((threshold-.5, threshold+.5),), origin=[0], scales=[1],
        max_mode_radius=1, seed=3, population_size=5, surrogate_budget=10,
        raster_method='master_block_mean_v1', source_resolution=16, input_resolution=8,
        material_fraction_bounds=(0, 1), lower_band=1, surrogate_contract_sha256='model',
        perturbation_offsets=(0.,))
    artifact = run_budgeted_nominal_feedback_proposal(state, source, ToyPredictor(),
        output_root=tmp_path/'feedback', provenance={'scope': 'toy'}, **kwargs)
    assert artifact['result'] == uncached
    assert json.loads((tmp_path/'feedback'/'proposal.json').read_bytes()) == artifact
    cost = json.loads((tmp_path/'feedback'/'cost.json').read_bytes())
    assert cost['inferences'] <= 10 and cost['cache_hits'] > 0
    assert audit_completed_inference_cost(tmp_path/'feedback'/'inferences', cost)['status'] == 'cost_consistent_not_source_authentication'
    assert audit_prediction_archive_against_calls(tmp_path/'feedback'/'inferences',
        tmp_path/'feedback'/'predictions.npz', cost)['status'] == 'archive_matches_calls_not_source_authentication'
    assert audit_nominal_prediction_archive(trace={'evaluations': artifact['result']['trace']},
        options={'perturbation_offsets': [0.0], 'raster_method': 'master_block_mean_v1',
                 'source_resolution': 16, 'input_resolution': 8, 'material_fraction_bounds': [0, 1],
                 'lower_band': 1, 'target_frequency': .5}, cost=cost,
        archive_path=tmp_path/'feedback'/'predictions.npz')['status'] == 'archive_consistent_not_model_authentication'
    with np.load(tmp_path/'feedback'/'predictions.npz', allow_pickle=False) as archive:
        assert len(archive.files) == 2 * cost['successful_inferences']
        for index in range(cost['successful_inferences']):
            assert np.array_equal(archive[f'output_{index:04d}'],
                ToyPredictor().predict_tensor(torch.as_tensor(archive[f'input_{index:04d}'])).numpy())
    with pytest.raises(FileExistsError):
        run_budgeted_nominal_feedback_proposal(state, source, ToyPredictor(),
            output_root=tmp_path/'feedback', provenance={'scope': 'toy'}, **kwargs)
    class Failed:
        def predict_tensor(self, raster):
            raise RuntimeError('inference failed')
    with pytest.raises(RuntimeError, match='inference failed'):
        run_budgeted_nominal_feedback_proposal(state, source, Failed(),
            output_root=tmp_path/'failure', provenance={'scope': 'toy'}, **kwargs)
    assert json.loads((tmp_path/'failure'/'cost.json').read_bytes())['inferences'] == 1
    pending = propose_feedback(state, **result['proposal'])
    assert pending.pending.parameter_hash == candidate.parameter_hash
    unknown = finish_feedback(pending, failure_reason='missing worker exit', failure_status='worker_result_unknown')
    with pytest.raises(ValueError, match='unknown worker result'):
        propose_fourier_candidate(unknown, source, ToyPredictor(), coefficient_keys=(),
            global_bounds=((threshold-.5, threshold+.5),), origin=[0], scales=[1],
            max_mode_radius=1, seed=4, population_size=5, surrogate_budget=10,
            raster_method='master_block_mean_v1', source_resolution=16, input_resolution=8,
            material_fraction_bounds=(0, 1), lower_band=1, surrogate_contract_sha256='model')
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


def test_projected_anchor_feedback_uses_gated_matching_physical_geometry():
    class ReversedPredictor(ToyPredictor):
        def predict_tensor(self, raster):
            return super().predict_tensor(raster).flip(-1)
    source=generate_unified_fourier_binary('source',master_seed=1,material_fraction_bin=1,
        config={'fourier_cutoff':2,'audit_resolution':16})
    threshold=source.parameters['threshold']
    offsets=(-1/16,0,1/16)
    grids=periodic_perturbation_grids(source,resolution=16,offsets=offsets)
    hashes=[hashlib.sha256(np.ascontiguousarray(g,dtype=np.uint8).tobytes()).hexdigest() for g in grids]
    observation={'row':{'lower_band_one_based':1,'refined_lower_edge_frequency':.2,
        'refined_upper_edge_frequency':.8,'last_refinement_stable':True,'resolution_stable':True,
        'adaptive_refinement_converged':True},
        'provenance':{'parameter_hash':source.parameter_hash,'comparison_contract_sha256':'solver',
        'offsets':list(offsets),'grid_sha256':hashes}}
    state=start_feedback(incumbent_hash=source.parameter_hash,incumbent_parameters=[threshold],
        incumbent_observation=observation,target_frequency=.5,radius=.1,min_radius=.01,
        max_radius=.2,candidate_budget=1)
    options=dict(coefficient_keys=(),global_bounds=((threshold-.5,threshold+.5),),
        origin=[0],scales=[1],max_mode_radius=1,seed=3,population_size=5,surrogate_budget=10,
        raster_method='master_block_mean_v1',source_resolution=16,input_resolution=8,
        material_fraction_bounds=(0,1),lower_band=1,surrogate_contract_sha256='a'*64,
        perturbation_offsets=offsets,projected_anchor_calibration=True)
    result=propose_fourier_candidate(state,source,ReversedPredictor(),**options)
    assert result['status']=='surrogate_proposal_only'
    assert result['edge_correction_applied']
    assert result['anchor_projection']['original_status']=='band_order_violation'
    assert result['proposal']['predicted_center']==pytest.approx(1.2)
    eligible=[row for row in result['trace'] if row['status']=='projected_surrogate_only']
    assert eligible and all(row['projected_prediction']['original_status']=='band_order_violation'
                            for row in eligible)
    assert result['anchor_edge_residual'] is not None
    class ConstantReversed:
        def predict_tensor(self,raster):
            return torch.tensor([.8,.2],dtype=raster.dtype).expand(len(raster),1,2)
    no_improvement=propose_fourier_candidate(state,source,ConstantReversed(),**options)
    assert no_improvement['status']=='no_predicted_improvement'
    assert no_improvement['anchor_projection']['original_status']=='band_order_violation'
    assert no_improvement['anchor_edge_residual']==pytest.approx([0,0],abs=1e-7)
    changed=observation.copy()
    changed['provenance']=dict(observation['provenance'],grid_sha256=['wrong']*3)
    invalid=start_feedback(incumbent_hash=source.parameter_hash,incumbent_parameters=[threshold],
        incumbent_observation=changed,target_frequency=.5,radius=.1,min_radius=.01,
        max_radius=.2,candidate_budget=1)
    with pytest.raises(ValueError,match='geometry mismatch'):
        propose_fourier_candidate(invalid,source,ReversedPredictor(),**options)
