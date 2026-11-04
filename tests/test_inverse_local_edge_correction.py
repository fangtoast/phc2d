"""Tests for solver-gated local band-edge correction.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import copy
import hashlib
import numpy as np
import pytest

from phc2d.inverse.local_edge_correction import fit_composite_edge_correction, fit_local_edge_correction, correct_projected_with_anchor
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids, project_perturbed_band_order
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary


def observation(identity, point, residual, *, stable=True, contract='c'):
    return {'parameter_hash': identity, 'parameters': [point], 'predicted_edges': [.4, .7],
            'physical_edges': [.4 + residual, .7 - residual],
            'numerical_gate_passed': stable, 'comparison_contract_sha256': contract,
            'surrogate_contract_sha256': 'surrogate'}


def test_local_residual_fit_recovers_affine_edge_bias_and_rejects_extrapolation():
    records = [observation('a', -.5, -.04), observation('b', 0., .01), observation('c', .5, .06)]
    model = fit_local_edge_correction(records, center=[0.], radius=1., ridge=1e-10)
    assert model.correct([.25], [.4, .7]) == pytest.approx([.435, .665], abs=1e-8)
    with pytest.raises(ValueError, match='outside'):
        model.correct([1.1], [.4, .7])


def test_local_correction_rejects_unstable_and_mixed_protocols():
    with pytest.raises(ValueError, match='numerical gates'):
        fit_local_edge_correction([observation('a', 0, .1, stable=False)], center=[0], radius=1, ridge=.01)
    with pytest.raises(ValueError, match='comparison contract'):
        fit_local_edge_correction([observation('a', 0, .1), observation('b', .1, .2, contract='d')],
                                  center=[0], radius=1, ridge=.01)
    other = observation('b', .1, .2)
    other['surrogate_contract_sha256'] = 'different'
    with pytest.raises(ValueError, match='surrogate contract'):
        fit_local_edge_correction([observation('a', 0, .1), other], center=[0], radius=1, ridge=.01)


def test_single_observation_gives_constant_correction():
    model = fit_local_edge_correction([observation('a', 0, .02)], center=[0], radius=1, ridge=.01)
    assert np.allclose(model.correct([.5], [.3, .8]), [.32, .78])


def test_composite_correction_uses_constant_when_points_are_collinear():
    rows = []
    for index, x in enumerate((-0.5, 0., 0.5)):
        row = observation(str(index), x, 0.02 + 0.01*x)
        row['parameters'] = [x, 0.0]
        rows.append(row)
    model = fit_composite_edge_correction(rows, center=[0, 0], radius=1., ridge=1e-6)
    assert np.all(model.coefficients[1:] == 0)
    assert model.correct([0.1, 0.1], [.4, .7]) == pytest.approx([.42, .68])
    rows[2]['parameters'][1] = 0.2
    model = fit_composite_edge_correction(rows, center=[0, 0], radius=1., ridge=1e-6)
    assert np.any(model.coefficients[1:] != 0)


def projected_case_inputs():
    source=generate_unified_fourier_binary('anchor',master_seed=1,material_fraction_bin=1,
        config={'fourier_cutoff':2,'audit_resolution':16})
    candidate=update_fourier_binary(source,sample_id='candidate',coefficients={},
        threshold=source.parameters['threshold']+.1,max_mode_radius=1)
    offsets=(-1/16,0,1/16)
    def projected(structure,bands):
        grids=periodic_perturbation_grids(structure,resolution=16,offsets=offsets)
        hashes=[hashlib.sha256(np.ascontiguousarray(g,dtype=np.uint8).tobytes()).hexdigest() for g in grids]
        return project_perturbed_band_order({'status':'band_order_violation',
            'bands':[[bands] for _ in offsets],'grid_sha256':hashes},lower_band=1,target_frequency=.5,
            surrogate_contract_sha256='a'*64)
    anchor=projected(source,[.8,.2])
    proposal=projected(candidate,[.7,.25])
    physical={'row':{'lower_band_one_based':1,'refined_lower_edge_frequency':.3,
        'refined_upper_edge_frequency':.9,'last_refinement_stable':True,'resolution_stable':True,
        'adaptive_refinement_converged':True},
        'provenance':{'parameter_hash':source.parameter_hash,'comparison_contract_sha256':'b'*64,
        'offsets':list(offsets),'grid_sha256':anchor['grid_sha256']}}
    return source,candidate,anchor,proposal,physical


def test_projected_anchor_correction_preserves_original_failure_and_binds_geometry():
    source,candidate,anchor,proposal,physical=projected_case_inputs()
    result=correct_projected_with_anchor(anchor,proposal,anchor_structure=source,
        candidate_structure=candidate,verified_anchor=physical,raster_resolution=16)
    assert result['status']=='calibrated_projected_surrogate_only'
    assert result['anchor_original_status']=='band_order_violation'
    assert result['candidate_original_status']=='band_order_violation'
    assert result['anchor_edge_residual']==pytest.approx([.1,.1])
    assert result['corrected_edges']==pytest.approx([.35,.8])
    assert result['corrected_target_margin']==pytest.approx(.6)


def test_projected_anchor_correction_rejects_tampering_and_failed_physical_gate():
    source,candidate,anchor,proposal,physical=projected_case_inputs()
    args=dict(anchor_structure=source,candidate_structure=candidate,verified_anchor=physical,raster_resolution=16)
    wrong=copy.deepcopy(proposal)
    wrong['gap']['lower_edge']+=.01
    with pytest.raises(ValueError,match='projection evidence'):
        correct_projected_with_anchor(anchor,wrong,**args)
    wrong=copy.deepcopy(proposal)
    wrong['surrogate_contract_sha256']='c'*64
    with pytest.raises(ValueError,match='surrogate contracts'):
        correct_projected_with_anchor(anchor,wrong,**args)
    broken=copy.deepcopy(physical)
    broken['row']['resolution_stable']=False
    with pytest.raises(ValueError,match='numerical gate'):
        correct_projected_with_anchor(anchor,proposal,**dict(args,verified_anchor=broken))
    with pytest.raises(ValueError,match='geometry mismatch'):
        correct_projected_with_anchor(anchor,proposal,**dict(args,candidate_structure=source))
