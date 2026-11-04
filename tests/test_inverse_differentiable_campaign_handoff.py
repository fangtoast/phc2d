"""Test that paired diagnostics retain roles while sharing identical MPB grids."""
import copy
import hashlib
import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.differentiable_campaign_handoff import build_campaign_request


def test_campaign_handoff_deduplicates_geometry_and_validates_hash():
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    grid = np.ascontiguousarray(source.rasterize((64, 64)), dtype=np.uint8)
    candidate = {'status': 'surrogate_only', 'structure': source.to_mapping(),
                 'parameter_hash': source.parameter_hash, 'master_grid_sha256': hashlib.sha256(grid.tobytes()).hexdigest()}
    points = [[0., 0.]]
    base = {'source_resolution': 64, 'k_points_sha256': hashlib.sha256(np.asarray(points, dtype=np.float32).tobytes()).hexdigest(),
            'physical_protocol': {'offsets': [0.0], 'raster_resolution': 64},
            'lower_band': 1, 'target_frequency': 0.5, 'solver_config': {}}
    manifest = {'base_config_sha256': 'base', 'trials': [{'trial_id': 'trial', 'source_id': 8}]}
    lock = {'scope': 'locked_paired_surrogate_campaign', 'manifest_sha256': 'manifest', 'k_points': points,
            'records': [{'trial_id': 'trial', 'role': 'evaluation_1', 'candidate': candidate},
                        {'trial_id': 'trial', 'role': 'top1', 'candidate': candidate}]}
    args = dict(lock=lock, manifest=manifest, base=base, lock_sha256='lock', manifest_sha256='manifest', base_sha256='base')
    physical, protocol = build_campaign_request(**args)
    assert len(physical['rows']) == 1 and len(physical['role_coverage']) == 2
    assert [row['case_index'] for row in physical['role_coverage']] == [0, 0]
    assert protocol['offsets'] == [0.0]
    bad = copy.deepcopy(lock)
    bad['records'][0]['candidate']['master_grid_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='structure hash'):
        build_campaign_request(**{**args, 'lock': bad})
