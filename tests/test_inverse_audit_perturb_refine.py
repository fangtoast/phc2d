"""Contract tests for the finite-offset refinement audit."""

import hashlib
import json

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_perturb_refine import run_case
from phc2d.inverse.audit_perturb_reconcile import reconcile_perturbations


def test_refinement_preserves_all_offset_grids_and_gates(tmp_path):
    structure = generate_unified_fourier_binary('s', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    manifest = {'rows': [{'structure': structure.to_mapping(), 'parameter_hash': structure.parameter_hash,
                          'source_index': 3, 'label': 'anchor'}], 'k_points': [[0, 0], [.25, 0]],
                'lower_band': 1, 'target_frequency': .5, 'solver_config': {'num_bands': 2, 'resolution': 16, 'polarization': 'TE',
                'eigensolver_tolerance': 1e-7, 'epsilon_air': structure.epsilon_air,
                'epsilon_material': structure.epsilon_material}}
    protocol = {'raster_resolution': 16, 'offsets': [-.0625, 0, .0625], 'refinement': {
        'verification_resolution': 24, 'initial_spacing': .25, 'min_levels': 2, 'max_levels': 2,
        'required_stable_levels': 2, 'edge_tolerance': 1, 'resolution_tolerance': 1e-6}}
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding='utf8')
        return path, hashlib.sha256(path.read_bytes()).hexdigest()
    mp, mh = save('manifest.json', manifest)
    pp, ph = save('protocol.json', protocol)
    def solver(grid, *, k_points, config):
        digest = hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest()
        return np.tile([.2, .7], (len(k_points), 1)), {'grid_sha256': digest}
    args = dict(expected_sha256=mh, protocol_path=pp, expected_protocol_sha256=ph,
                case_index=0, output_root=tmp_path / 'out', solver=solver)
    result = json.loads((run_case(mp, **args) / 'results.json').read_text())
    assert [r['offset'] for r in result['observations']] == protocol['offsets']
    assert len(result['aggregate']['provenance']['grid_sha256']) == 3
    assert result['aggregate']['status'] == 'numerically_stable'
    assert result['aggregate']['row']['refined_lower_edge_frequency'] == pytest.approx(.2)
    summary = reconcile_perturbations(mp, [result], manifest_sha256=mh, protocol_path=pp, protocol_sha256=ph)
    assert summary['numerical_gate_passed_count'] == 1
    result['observations'][0]['provenance']['initial_solver']['grid_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='grid identity'):
        reconcile_perturbations(mp, [result], manifest_sha256=mh, protocol_path=pp, protocol_sha256=ph)
    with pytest.raises(FileExistsError):
        run_case(mp, **args)
