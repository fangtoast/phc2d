"""Verify solver-to-result identity and genuinely unseen perturbation contracts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_perturb_refine import run_case
from phc2d.inverse.design_evidence import RobustSolverArtifact, bind_joint_robust_evidence
from phc2d.inverse.design_metrics import summarize_design_outcomes
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def make_artifact(tmp_path):
    source = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    candidate = update_fourier_binary(source, sample_id='candidate', coefficients={},
                                     threshold=source.parameters['threshold']+.1, max_mode_radius=1)
    config = FullCellMpbBandSolverConfig(num_bands=2, resolution=16, polarization='TE',
        epsilon_air=source.epsilon_air, epsilon_material=source.epsilon_material, eigensolver_tolerance=1e-7)
    refinement = {'verification_resolution': 24, 'initial_spacing': .25, 'min_levels': 2,
                  'max_levels': 2, 'required_stable_levels': 2, 'edge_tolerance': 1,
                  'resolution_tolerance': 1e-6}
    manifest = {'k_points': [[0., 0.], [.25, 0.]], 'lower_band': 1, 'target_frequency': .5,
                'solver_config': asdict(config), 'rows': [{'source_index': 3, 'label': 'candidate',
                'parameter_hash': candidate.parameter_hash, 'structure': candidate.to_mapping()}]}
    serial = 0
    def make(offsets, *, unstable=False, alternate_design=False):
        nonlocal serial
        folder = tmp_path / str(serial)
        serial += 1
        folder.mkdir()
        protocol = {'offsets': offsets, 'raster_resolution': 16, 'refinement': refinement}
        mp, pp = folder/'manifest.json', folder/'protocol.json'
        current_manifest = json.loads(json.dumps(manifest))
        if alternate_design:
            other = update_fourier_binary(source, sample_id='other', coefficients={},
                threshold=source.parameters['threshold']+.2, max_mode_radius=1)
            current_manifest['rows'][0]['parameter_hash'] = other.parameter_hash
            current_manifest['rows'][0]['structure'] = other.to_mapping()
        mp.write_text(json.dumps(current_manifest), encoding='utf8')
        pp.write_text(json.dumps(protocol), encoding='utf8')
        def solver(grid, *, k_points, config):
            fill = float(grid.mean())
            extra = .01 if unstable and config.resolution == 24 else 0.
            metadata = {'grid_sha256': digest(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()),
                        'resolution': config.resolution, 'polarization': config.polarization,
                        'num_bands': config.num_bands, 'mesh_size': config.mesh_size,
                        'band_guard_count': config.band_guard_count, 'band_order_policy': config.band_order_policy,
                        'eigensolver_tolerance': config.eigensolver_tolerance, 'epsilon_air': config.epsilon_air,
                        'epsilon_material': config.epsilon_material, 'solver_backend': 'mock', 'solver_version': '1',
                        'frequency_normalization': 'synthetic unit'}
            return np.tile([.2+.1*fill+extra, .8-.1*fill-extra], (len(k_points), 1)), metadata
        mh, ph = digest(mp.read_bytes()), digest(pp.read_bytes())
        output = run_case(mp, expected_sha256=mh, protocol_path=pp, expected_protocol_sha256=ph,
                          case_index=0, output_root=folder/'cases', solver=solver)
        raw = (output/'results.json').read_bytes()
        return RobustSolverArtifact(mp.read_bytes(), mh, pp.read_bytes(), ph, raw, digest(raw), 0)
    return make


def bind(nominal, unseen):
    return bind_joint_robust_evidence(nominal, unseen,
        expected_solver_identity={'solver_backend': 'mock', 'solver_version': '1'},
        expected_frequency_normalization='synthetic unit')


def test_joint_solver_evidence_checks_raw_records_and_intersects_intervals(make_artifact):
    nominal = make_artifact((-.0625, 0., .0625))
    unseen = make_artifact((-.125, 0., .125))
    pair = bind(nominal, unseen)
    assert pair['nominal']['status'] == pair['unseen']['status'] == 'verified'
    assert pair['nominal']['design_sha256'] == pair['unseen']['design_sha256']
    expected = [{'target_id': 'a', 'seed': 0, 'target_frequency': .5}]
    summary = summarize_design_outcomes(expected, [{'target_id': 'a', 'seed': 0,
        'target_frequency': .5, 'nominal': pair['nominal'], 'unseen': pair['unseen']}])
    assert summary['counts']['both_verified'] == summary['counts']['both_success'] == 1
    assert summary['mean_successful_joint_common_width'] < summary['mean_successful_nominal_common_width']
    with pytest.raises(ValueError, match='result digest mismatch'):
        bind(nominal, replace(unseen, result_bytes=unseen.result_bytes+b' '))


def test_unseen_offsets_and_grids_cannot_reuse_optimization_cases(make_artifact):
    nominal = make_artifact((-.0625, 0., .0625))
    with pytest.raises(ValueError, match='held-out offsets'):
        bind(nominal, nominal)
    same_grid = make_artifact((-.01, 0., .01))
    with pytest.raises(ValueError, match='held-out grids'):
        bind(nominal, same_grid)
    other = make_artifact((-.125, 0., .125), alternate_design=True)
    with pytest.raises(ValueError, match='different base designs'):
        bind(nominal, other)


def test_unstable_solver_remains_unresolved_and_normalization_is_checked(make_artifact):
    nominal = make_artifact((-.0625, 0., .0625))
    unstable = make_artifact((-.125, 0., .125), unstable=True)
    pair = bind(nominal, unstable)
    assert pair['unseen']['status'] == 'unresolved'
    assert 'lower_edge' not in pair['unseen']
    with pytest.raises(ValueError, match='frequency normalization'):
        bind_joint_robust_evidence(nominal, unstable,
            expected_solver_identity={'solver_backend': 'mock', 'solver_version': '1'},
            expected_frequency_normalization='wrong unit')
