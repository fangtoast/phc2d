"""Check independent targets and seeds retain frozen source and method identities."""
import copy

import pytest

from phc2d.inverse import independent_campaign as campaign


def test_independent_matrix_pairs_each_target_and_seed(monkeypatch):
    def build(spec, base, sources):
        return {'scope': 'preregistered_surrogate_design', 'trials': [], 'sources': list(sources)}
    monkeypatch.setattr(campaign, 'build_campaign', build)
    roster = {'scope': 'heldout_inverse_source_selection_no_band_labels', 'method_lock_sha256': 'method',
              'sources': [{'id': 100003, 'sha256': 'source', 'split': 'test'}]}
    settings = {'experiment_id': 'EXP-INV-016', 'method_lock_sha256': 'method', 'source_roster_sha256': 'roster',
        'target_frequencies': [.55, .57], 'optimization_seeds': [1, 2], 'methods': ['adam', 'de', 'random'],
        'evaluations': 60, 'initial_options': [[[0., 0.]]], 'diagnostic_evaluation_counts': [1, 20, 40, 60]}
    matrix = campaign.build_independent_matrix(roster, {100003: object()}, base={'target_frequency': .57},
        settings=settings, roster_sha256='roster', method_lock_sha256='method')
    assert len(matrix) == 4
    assert {(row['target_frequency'], row['seed']) for row in matrix} == {(.55, 1), (.55, 2), (.57, 1), (.57, 2)}
    assert all(row['manifest']['method_lock_sha256'] == 'method' for row in matrix)
    assert matrix[0]['base']['target_frequency'] == .55
    bad = copy.deepcopy(roster); bad['sources'][0]['split'] = 'train'
    with pytest.raises(ValueError, match='test-only'):
        campaign.build_independent_matrix(bad, {100003: object()}, base={}, settings=settings,
                                          roster_sha256='roster', method_lock_sha256='method')
