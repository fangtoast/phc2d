"""Tests for paired initial preparation and fallback ordering."""
from types import SimpleNamespace
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse import differentiable_campaign


def test_initial_fallback_keeps_attempts_and_shared_point(monkeypatch):
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    statuses = iter(['material_fraction_rejected', 'eligible'])
    monkeypatch.setattr(differentiable_campaign, 'screen_fourier_geometry',
                        lambda *a, **k: SimpleNamespace(status=next(statuses), material_fraction=0.5))
    base = {'coefficient_keys': [[0, 'cos']], 'scales': [0.2, 0.1],
            'lower': [-0.5, -0.5], 'upper': [0.5, 0.5], 'max_mode_radius': 1.0,
            'source_resolution': 64, 'input_resolution': 16, 'material_fraction_bounds': [0.5, 0.9]}
    rows = differentiable_campaign.prepare_paired_initials(source, base=base,
        campaign={'initial_options': [[[0.15, 0.0], [-0.15, 0.0]]]})
    assert rows[0]['initial'] == [-0.15, 0.0]
    assert [item['status'] for item in rows[0]['attempts']] == ['material_fraction_rejected', 'eligible']


def test_campaign_pairs_initials_and_preserves_missing_cases(monkeypatch):
    monkeypatch.setattr(differentiable_campaign, 'prepare_paired_initials', lambda source, **kw: [
        {'initial_index': 0, 'status': 'ready', 'initial': [0.0, 0.0], 'attempts': []},
        {'initial_index': 1, 'status': 'ready', 'initial': [0.15, 0.0], 'attempts': []},
        {'initial_index': 2, 'status': 'no_valid_initial', 'initial': None, 'attempts': []}])
    campaign = {'experiment_id': 'EXP-INV-013', 'sources': [{'id': i, 'sha256': str(i)} for i in (8, 491, 348)],
                'methods': ['adam', 'de', 'random'], 'evaluations': 60, 'seed': 7}
    sources = {i: SimpleNamespace(parameter_hash=str(i)) for i in (8, 491, 348)}
    result = differentiable_campaign.build_campaign(campaign, {}, sources)
    assert len(result['trials']) == 27
    assert len({row['trial_id'] for row in result['trials']}) == 27
    for source_id in sources:
        paired = [row for row in result['trials'] if row['source_id'] == source_id and row['initial_index'] == 1]
        assert [row['config']['initial'] for row in paired] == [[0.15, 0.0]] * 3
        assert all(row['config']['evaluations'] == 60 for row in paired)
    assert sum(row['config'] is None for row in result['trials']) == 9
    with pytest.raises(ValueError, match='roster'):
        differentiable_campaign.build_campaign(campaign, {}, {8: sources[8]})
