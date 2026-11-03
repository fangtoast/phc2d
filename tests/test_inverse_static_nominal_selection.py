"""Contract checks for static surrogate candidate selection.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json
import numpy as np

import pytest

torch = pytest.importorskip('torch')

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.inverse.surrogate_design import (audit_surrogate_design, run_surrogate_design,
                                            run_budgeted_nominal_static_design, audit_nominal_prediction_archive)
from phc2d.inverse.trace_selection import lock_static_nominal_candidates
from phc2d.inverse.frozen_band_adapter import SingleSampleBudgetedPredictor, audit_completed_inference_cost


def test_static_selection_deduplicates_grids_and_breaks_ties_by_trace_order():
    source = generate_unified_fourier_binary('anchor', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    def make(delta, name):
        return update_fourier_binary(source, sample_id=name, coefficients={},
            threshold=source.parameters['threshold'] + delta, max_mode_radius=1)
    first, second = make(-.2, 'first'), make(.2, 'second')
    def row(index, item, score):
        return {'candidate_index': index, 'status': 'surrogate_only', 'structure': item.to_mapping(),
                'parameter_hash': item.parameter_hash, 'selection_score': score}
    traces = [[row(0, source, 100), row(1, first, .4)], [row(0, first, .9), row(1, second, .4)]]
    selected = lock_static_nominal_candidates(source, traces, raster_resolution=32,
                                               candidates=2, evaluations_per_trace=2)
    assert selected['status'] == 'locked_surrogate_only'
    assert [(r['trace_index'], r['candidate_index']) for r in selected['selected']] == [(0, 1), (1, 1)]
    assert selected['excluded_duplicates'] == [{'trace_index': 0, 'candidate_index': 0},
                                                {'trace_index': 1, 'candidate_index': 0}]
    assert selected['physical_validation'] == 'not_run'
    assert lock_static_nominal_candidates(source, traces, raster_resolution=32, candidates=3,
        evaluations_per_trace=2)['selected'] == []
    traces[1][1]['perturbation_offsets'] = [-.1, 0., .1]
    with pytest.raises(ValueError, match='not nominal'):
        lock_static_nominal_candidates(source, traces, raster_resolution=32, candidates=2, evaluations_per_trace=2)
    del traces[1][1]['perturbation_offsets']
    traces[1][1]['parameter_hash'] = 'tampered'
    with pytest.raises(ValueError, match='parameter identity'):
        lock_static_nominal_candidates(source, traces, raster_resolution=32, candidates=2, evaluations_per_trace=2)


def test_two_audited_static_searches_feed_locked_selection(tmp_path):
    class Predictor:
        def predict_tensor(self, raster):
            mean = raster.mean(dim=(1, 2))
            return torch.stack((.2 + .1 * mean, .8 - .1 * mean), dim=1).unsqueeze(1)

    source = generate_unified_fourier_binary('pilot', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    threshold = source.parameters['threshold']
    options = {'coefficient_keys': (), 'bounds': ((threshold-.2, threshold+.2),),
        'max_mode_radius': 1, 'population_size': 5, 'evaluation_budget': 10,
        'raster_method': 'master_block_mean_v1', 'source_resolution': 16,
        'input_resolution': 8, 'material_fraction_bounds': (0, 1),
        'lower_band': 1, 'target_frequency': .5}
    seeds = [2026110501, 2026110502]
    source_raw = json.dumps(source.to_mapping()).encode()
    config_raw = json.dumps({'search_options': options, 'seeds': seeds, 'checkpoint_sha256': 'a'*64}).encode()
    provenance = {'source_sha256': hashlib.sha256(source_raw).hexdigest(),
        'config_sha256': hashlib.sha256(config_raw).hexdigest(), 'checkpoint_sha256': 'a'*64}
    out = tmp_path/'static'
    run_surrogate_design(source, Predictor(), seeds=seeds, search_options=options,
                         output_root=out, provenance=provenance)
    assert audit_surrogate_design(out, config_bytes=config_raw, source_bytes=source_raw,
                                  checkpoint_sha256='a'*64)['status'] == 'artifact_consistent'
    traces = [json.loads((out/f'seed_{seed}'/'search.json').read_bytes())['evaluations'] for seed in seeds]
    locked = lock_static_nominal_candidates(source, traces, raster_resolution=16,
                                             candidates=2, evaluations_per_trace=10)
    assert locked['status'] == 'locked_surrogate_only'
    assert locked['physical_validation'] == 'not_run'
    assert len({row['nominal_grid_sha256'] for row in locked['selected']}) == 2
    assert locked['selected'][0]['selection_score'] >= locked['selected'][1]['selection_score']


def test_static_selected_spectrum_reuses_prior_inference_within_budget(tmp_path):
    class Predictor:
        def predict_tensor(self, raster):
            mean = raster.mean(dim=(1, 2))
            return torch.stack((.2 + .1 * mean, .8 - .1 * mean), dim=1).unsqueeze(1)

    source = generate_unified_fourier_binary('pilot', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    threshold = source.parameters['threshold']
    options = {'coefficient_keys': (), 'bounds': ((threshold-.2, threshold+.2),),
        'max_mode_radius': 1, 'population_size': 5, 'evaluation_budget': 10,
        'raster_method': 'master_block_mean_v1', 'source_resolution': 16, 'input_resolution': 8,
        'material_fraction_bounds': (0, 1), 'lower_band': 1, 'target_frequency': .5,
        'perturbation_offsets': (0.,)}
    cached = SingleSampleBudgetedPredictor(Predictor(), max_inferences=10, cache_capacity=10)
    run_surrogate_design(source, cached, seeds=[2026110501], search_options=options,
                         output_root=tmp_path/'cached', provenance={'scope': 'toy'})
    run_surrogate_design(source, Predictor(), seeds=[2026110501], search_options=options,
                         output_root=tmp_path/'reference', provenance={'scope': 'toy'})
    assert cached.inferences <= 10
    for name in ('search.json', 'predicted_response.json'):
        assert json.loads((tmp_path/'cached'/'seed_2026110501'/name).read_bytes()) == json.loads(
            (tmp_path/'reference'/'seed_2026110501'/name).read_bytes())
    source_raw = json.dumps(source.to_mapping()).encode()
    config_raw = json.dumps({'search_options': options, 'seeds': [2026110501, 2026110502],
                             'checkpoint_sha256': 'a'*64}).encode()
    provenance = {'source_sha256': hashlib.sha256(source_raw).hexdigest(),
                  'config_sha256': hashlib.sha256(config_raw).hexdigest(), 'checkpoint_sha256': 'a'*64}
    paired = run_budgeted_nominal_static_design(source, Predictor(), seeds=[2026110501, 2026110502],
        search_options=options, output_root=tmp_path/'paired', provenance=provenance, config_bytes=config_raw)
    assert len(paired['trajectories']) == 2
    for record in paired['trajectories']:
        assert record['inferences'] <= 10 and record['max_inferences'] == 10
        assert json.loads((tmp_path/'paired'/f"cost_{record['seed']}.json").read_bytes()) == record
        assert audit_completed_inference_cost(tmp_path/'paired'/f"inference_{record['seed']}", record)['status'] == 'cost_consistent_not_source_authentication'
        with pytest.raises(ValueError, match='cost record and inference evidence'):
            audit_completed_inference_cost(tmp_path/'paired'/f"inference_{record['seed']}",
                                           {**record, 'inferences': record['inferences'] - 1})
        with np.load(tmp_path/'paired'/f"predictions_{record['seed']}.npz", allow_pickle=False) as archive:
            assert len(archive.files) == 2 * record['successful_inferences']
            for index in range(record['successful_inferences']):
                x = torch.as_tensor(archive[f'input_{index:04d}'])
                assert np.array_equal(archive[f'output_{index:04d}'], Predictor().predict_tensor(x).numpy())
        assert audit_surrogate_design(tmp_path/'paired'/f"trajectory_{record['seed']}",
            config_bytes=(tmp_path/'paired'/f"config_{record['seed']}.json").read_bytes(),
            source_bytes=source_raw, checkpoint_sha256='a'*64)['status'] == 'artifact_consistent'
        trace = json.loads((tmp_path/'paired'/f"trajectory_{record['seed']}"/f"seed_{record['seed']}"/'search.json').read_bytes())
        assert audit_nominal_prediction_archive(trace=trace, options=options, cost=record,
            archive_path=tmp_path/'paired'/f"predictions_{record['seed']}.npz")['status'] == 'archive_consistent_not_model_authentication'
        bad_trace = json.loads(json.dumps(trace))
        bad_trace['evaluations'][0]['predicted_gap']['target_margin'] += .01
        with pytest.raises(ValueError, match='candidate score differ'):
            audit_nominal_prediction_archive(trace=bad_trace, options=options, cost=record,
                archive_path=tmp_path/'paired'/f"predictions_{record['seed']}.npz")
    with pytest.raises(FileExistsError):
        run_budgeted_nominal_static_design(source, Predictor(), seeds=[2026110501, 2026110502],
            search_options=options, output_root=tmp_path/'paired', provenance=provenance, config_bytes=config_raw)

    class Failed:
        def predict_tensor(self, raster):
            raise RuntimeError('inference failed')
    with pytest.raises(RuntimeError, match='inference failed'):
        run_budgeted_nominal_static_design(source, Failed(), seeds=[2026110501, 2026110502],
            search_options=options, output_root=tmp_path/'failed', provenance=provenance, config_bytes=config_raw)
    failure = json.loads((tmp_path/'failed'/'cost_2026110501.json').read_bytes())
    assert failure['status'] == 'failed' and failure['inferences'] == 1
    assert not (tmp_path/'failed'/'cost_2026110502.json').exists()
