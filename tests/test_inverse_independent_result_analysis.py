"""Independent comparisons pair roles and aggregate source-level effects."""
import pytest

from phc2d.inverse.independent_result_analysis import analyze_independent


def _report(source, values, *, feedback=False):
    suffix = '/final' if feedback else '/top1'
    prefix = f'source_{source}_initial_0_' + ('adam_' if feedback else '')
    return {'scope':'locked_robust_surrogate_vs_physical','roles':[
        {'label':prefix+method+suffix,'status':'numerically_stable',
         'physical_common_target_margin':value,'physical_nominal_target_margin':value,
         'physical_contains_target':value>0} for method,value in values.items()]}


def test_pairs_independent_methods_and_reports_source_level_interaction():
    pure={'target_00_seed_1':_report('008',{'adam':.1,'de':.0,'random':-.1})}
    feedback={'target_00_seed_1':_report('008',{'adam_trajectory':.12,'adam_residual':.15,'de_residual':.02},feedback=True)}
    result=analyze_independent(pure,feedback)
    assert result['groups']['adam_residual']['common_target_covered']==1
    assert result['independent_source_count']==1
    assert result['comparisons']['residual_minus_trajectory']['mean_of_source_means']==pytest.approx(.03)
    assert result['gradient_feedback_interaction']['mean_of_source_means']==pytest.approx(.03)
    assert result['by_target_index']['00']['adam']['registered']==1
    assert result['by_source']['008']['adam']['registered']==1
    feedback['target_00_seed_1']['roles'].append(feedback['target_00_seed_1']['roles'][0])
    with pytest.raises(ValueError,match='duplicate'):
        analyze_independent(pure,feedback)
