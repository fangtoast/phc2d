"""Exercise the registered feedback stencil and boundary coverage."""
import pytest
torch = pytest.importorskip('torch')

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.differentiable_feedback_seed import prepare_feedback_stencil
from phc2d.inverse.differentiable_robust_stencil import predict_stencil_scenarios
from phc2d.inverse.differentiable_robust_feedback import (bind_scenario_feedback,
    bind_robust_stencil_batch, fit_scenario_corrections)
from phc2d.inverse.differentiable_robust_corrected import propose_corrected_robust_design
from phc2d.inverse.differentiable_robust_stencil_lock import lock_robust_stencils


def test_feedback_stencil_retains_all_roles_and_bounds():
    class Predictor:
        device = torch.device('cpu')
        checkpoint_sha256 = 'checkpoint'
        def predict_tensor(self, image):
            value = image.mean(dim=(1, 2))
            return torch.stack((0.2 + 0.1 * value, 0.7 + 0.1 * value), dim=1).unsqueeze(1)
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    base = {'coefficient_keys': [[0, 'cos']], 'scales': [0.2, 0.1], 'lower': [-0.5, -0.5],
            'upper': [0.5, 0.5], 'max_mode_radius': 1.0, 'source_resolution': 64,
            'input_resolution': 16, 'material_fraction_bounds': [0, 1], 'lower_band': 1,
            'target_frequency': .5, 'temperature': .2, 'beta': 100.0}
    cfg = {'initial_offsets': [[0, 0], [.1, 0], [-.1, 0], [0, .1], [0, -.1]]}
    result = prepare_feedback_stencil(source, Predictor(), base=base, campaign=cfg, initial=[.5, 0])
    assert len(result['records']) == 5
    assert result['records'][1]['status'] == 'outside_parameter_bounds'
    assert result['records'][0]['point'] == [.5, 0]
    assert result['soft_audit']['roles'][1]['status'] == 'not_eligible'
    edges = result['records'][0]['surrogate_edges']
    assert edges['soft_smooth_edges'][0] >= edges['soft_exact_edges'][0]
    assert edges['soft_smooth_edges'][1] <= edges['soft_exact_edges'][1]
    scenarios = predict_stencil_scenarios(result, source, Predictor(), base=base, offsets=[-1/64, 0., 1/64])
    assert len(scenarios['records']) == 5 and scenarios['records'][1]['status'] == 'outside_parameter_bounds'
    row = scenarios['records'][0]
    assert row['status'] == 'computed' and len(row['hard']['bands']) == 3
    assert row['soft_smooth_edges'][1] == pytest.approx(edges['soft_smooth_edges'], abs=1e-6)
    assert row['hard_edges'][1] == pytest.approx(edges['hard_edges'], abs=1e-6)
    runtime_contract = {'precision': 'fp32', 'amp_enabled': False, 'tf32_enabled': False}
    name, trial_id = 'source_0008_initial_0', 'source_0008_initial_0_nominal'
    origin = {**result, 'design_manifest_sha256': 'parent', 'checkpoint_sha256': 'checkpoint', 'k_points_sha256': 'k'}
    scenario = {**scenarios, 'source_id': 8, 'initial_index': 0, 'manifest_sha256': 'manifest',
        'stencil_sha256': 'stencil', 'checkpoint_sha256': 'checkpoint', 'k_points_sha256': 'k', 'code_commit': 'commit',
        'cluster_id': 'n56', 'runtime': {'contract': runtime_contract, 'matmul_tf32_enabled': False,
                                         'cudnn_tf32_enabled': False}}
    manifest = {'scope': 'preregistered_robust_design', 'parent_manifest_sha256': 'parent',
        'offsets': [-1/64, 0., 1/64], 'sources': [{'source_id': 8, 'source_parameter_hash': source.parameter_hash,
        'initials': [{'initial_index': 0, 'status': 'ready', 'initial': [.5, 0.]}]}],
        'trials': [{'trial_id': trial_id, 'config': {'checkpoint_sha256': 'checkpoint', 'k_points_sha256': 'k',
                                                      'runtime': runtime_contract}}]}
    locked = lock_robust_stencils(manifest, {name: scenario}, {name: origin}, manifest_sha256='manifest',
        design_lock={'manifest_sha256': 'parent', 'k_points': [[0., 0.]]}, nominal_sha256={name: 'stencil'})
    assert len(locked['records']) == 5 and locked['records'][1]['status'] == 'outside_parameter_bounds'
    offsets = [-1/64, 0., 1/64]
    physical = {'case_index': 0, 'parameter_hash': row['parameter_hash'], 'manifest_sha256': 'manifest',
        'protocol_sha256': 'protocol', 'aggregate': {'status': 'numerically_stable'}, 'observations': []}
    for index, offset in enumerate(offsets):
        low, high = row['hard_edges'][index]
        physical['observations'].append({'offset': offset, 'provenance': {
            'evaluated_grid_sha256': row['hard']['grid_sha256'][index], 'parameter_hash': row['parameter_hash'],
            'comparison_contract_sha256': 'refinement'}, 'arrays': {'k_points': [[0., 0.]],
            'bands': row['hard']['bands'][index]}, 'row': {'refined_lower_edge_frequency': low,
            'refined_upper_edge_frequency': high, 'last_refinement_stable': True,
            'resolution_stable': True, 'adaptive_refinement_converged': True}})
    bound = bind_scenario_feedback(row, physical, expected_case_index=0, expected_parameter_hash=row['parameter_hash'],
        offsets=offsets, manifest_sha256='manifest', protocol_sha256='protocol', surrogate_contract_sha256='model',
        k_points=[[0., 0.]], lower_band=1, beta=100)
    batch_lock = {'purpose': 'robust_feedback_stencil', 'records': [{'label': 'trial/offset_0',
        'trial_id': 'trial', 'role': 'offset_0', 'candidate': result['records'][0]['candidate'],
        'scenario_prediction': row}]}
    batch_manifest = {'selection_lock_sha256': 'lock', 'protocol_sha256': 'protocol', 'k_points': [[0., 0.]],
        'role_coverage': [{'label': 'trial/offset_0', 'case_index': 0}], 'rows': [
            {'parameter_hash': row['parameter_hash'], 'master_grid_sha256': row['hard']['grid_sha256'][1]}]}
    batch = bind_robust_stencil_batch(batch_lock, batch_manifest, {'offsets': offsets}, {0: physical},
        base={**base, 'checkpoint_sha256': 'checkpoint', 'k_points_sha256': 'k'}, lock_sha256='lock',
        manifest_sha256='manifest', protocol_sha256='protocol')
    assert batch['records'][0]['status'] == 'observed' and len(batch['records'][0]['scenarios']) == 3
    corrections = fit_scenario_corrections(bound, offsets=offsets, center=[.5, 0.], radius=.2, ridge=1e-5)
    assert len(corrections) == 3 and all(item.sample_count == 1 for item in corrections)
    assert len({item.comparison_contract_sha256 for item in corrections}) == 3
    config = {**base, 'checkpoint_sha256': 'checkpoint', 'surrogate_contract_sha256': 'model',
              'offsets': offsets, 'center': [.5, 0.], 'radius': .2, 'ridge': 1e-5,
              'evaluations': 2, 'learning_rate': .05}
    proposal = propose_corrected_robust_design(source, Predictor(), bound, config=config)
    assert proposal['status'] == 'proposal_ready' and len(proposal['history']) == 2
    assert len(proposal['prefit_prediction']['corrected_smooth_edges']) == 3
    assert all(item['sample_count'] == 1 for item in proposal['corrections'])
    with pytest.raises(ValueError, match='scenario'):
        fit_scenario_corrections(bound[:-1], offsets=offsets, center=[.5, 0.], radius=.2, ridge=1e-5)
