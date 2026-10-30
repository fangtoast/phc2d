"""Contracts for isolated inverse refinement audit cases."""

import hashlib
import json

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_refine import run_case


def test_frozen_refinement_writes_numerical_evidence_and_refuses_overwrite(tmp_path):
    structure = generate_unified_fourier_binary('s', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    manifest = {'rows': [{'structure': structure.to_mapping(), 'parameter_hash': structure.parameter_hash,
                          'source_index': 1, 'label': 'nominal'}],
                'k_points': [[0, 0], [.25, 0]], 'lower_band': 1,
                'solver_config': {'num_bands': 2, 'resolution': 16, 'polarization': 'TE',
                                  'eigensolver_tolerance': 1e-7,
                                  'epsilon_air': structure.epsilon_air,
                                  'epsilon_material': structure.epsilon_material}}
    options = {'verification_resolution': 24, 'initial_spacing': .25, 'min_levels': 2,
               'max_levels': 2, 'required_stable_levels': 2, 'edge_tolerance': 1,
               'resolution_tolerance': 1e-6}
    def save(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data), encoding='utf8')
        return path, hashlib.sha256(path.read_bytes()).hexdigest()
    mp, mh = save('manifest.json', manifest)
    op, oh = save('options.json', options)
    def solver(item, *, k_points, config):
        r = np.sum(np.asarray(k_points) ** 2, axis=1)
        return np.column_stack((.2 + .01*r, .7 - .01*r)), {'geometry_hash': item.geometry_hash}
    args = dict(expected_sha256=mh, refinement_path=op, expected_refinement_sha256=oh,
                case_index=0, output_root=tmp_path / 'out', solver=solver)
    result = json.loads((run_case(mp, **args) / 'results.json').read_text())
    assert result['rows'][0]['adaptive_refinement_converged']
    assert result['rows'][0]['resolution_stable']
    assert result['provenance']['comparison_contract_sha256']
    assert len(result['arrays']['k_points']) > 2
    with pytest.raises(FileExistsError):
        run_case(mp, **args)
    with pytest.raises(ValueError, match='SHA-256'):
        run_case(mp, **{**args, 'expected_refinement_sha256': 'wrong'})
