"""Select held-out inverse structures without accessing any band labels."""
import hashlib
import json

import numpy as np
import pytest

h5py = pytest.importorskip('h5py')
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.independent_sources import (select_heldout_sources, write_heldout_sources,
    collect_inverse_exclusions, validate_frozen_method_lock)


def test_heldout_selection_preserves_split_and_development_exclusions(tmp_path):
    method = json.dumps({'scope': 'frozen_inverse_method_selection', 'status': 'frozen',
                         'selected_method': 'adam_residual', 'evidence_sha256': 'evidence'}).encode()
    assert validate_frozen_method_lock(method, hashlib.sha256(method).hexdigest())['status'] == 'frozen'
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_frozen_method_lock(method, 'wrong')
    structures = [generate_unified_fourier_binary(f'source-{i}', master_seed=12, material_fraction_bin=1,
        config={'fourier_cutoff': 3, 'audit_resolution': 32}) for i in range(5)]
    path = tmp_path/'dataset.h5'
    with h5py.File(path, 'w') as data:
        data.create_dataset('idx_train', data=[0])
        data.create_dataset('idx_val', data=[1])
        data.create_dataset('idx_test', data=[2, 3, 4])
        data.create_dataset('structure_json', data=[json.dumps(item.to_mapping()).encode() for item in structures])
        data.create_dataset('parameter_hash', data=[item.parameter_hash.encode() for item in structures])
    dataset_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    excluded = structures[2]
    grid = np.ascontiguousarray(excluded.rasterize((64, 64)), dtype=np.uint8)
    evidence = json.dumps({'nested': [{'candidate': {'parameter_hash': excluded.parameter_hash,
                            'master_grid_sha256': hashlib.sha256(grid.tobytes()).hexdigest()}}]}).encode()
    exclusions = collect_inverse_exclusions([{'name': 'prior', 'bytes': evidence,
                                              'sha256': hashlib.sha256(evidence).hexdigest()}])
    assert excluded.parameter_hash in exclusions['parameter_hashes']
    separate = json.dumps({'parameters': [{'parameter_hash': 'parameter-only'}],
                           'grids': [{'master_grid_sha256': 'grid-only'}]}).encode()
    separate_exclusions = collect_inverse_exclusions([{'name': 'separate', 'bytes': separate,
                                                       'sha256': hashlib.sha256(separate).hexdigest()}])
    assert separate_exclusions['parameter_hashes'] == ['parameter-only']
    assert separate_exclusions['master_grid_sha256'] == ['grid-only']
    kwargs = dict(dataset_sha256=dataset_sha, method_lock_sha256='method', seed=7, count=2,
        source_id_offset=100000, master_resolution=64, excluded_parameter_hashes=[excluded.parameter_hash],
        excluded_grid_hashes=[hashlib.sha256(grid.tobytes()).hexdigest()], epsilon_air=excluded.epsilon_air,
        epsilon_material=excluded.epsilon_material)
    selected = select_heldout_sources(path, **kwargs)
    assert {row['dataset_index'] for row in selected['rows']} == {3, 4}
    assert all(row['split'] == 'test' for row in selected['rows'])
    roster = write_heldout_sources(selected, tmp_path/'sources')
    assert {row['id'] for row in roster['sources']} == {100003, 100004}
    for row in roster['sources']:
        raw = (tmp_path/'sources'/f"source_{row['id']:04d}.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['sha256']
    with pytest.raises(ValueError, match='dataset identity'):
        select_heldout_sources(path, **{**kwargs, 'dataset_sha256': 'wrong'})
