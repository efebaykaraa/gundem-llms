import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.scoring import score_result, score_report
from lab.evaluate import summarize


def result(quality):
    return dict(model_id='one', variant='distilled', exact_match=quality,
                topic_micro_f1=quality, type_accuracy=quality,
                score_mae={k: 3 * (1 - quality) for k in ('etki', 'ciddiyet', 'aciliyet')})


@pytest.mark.parametrize('quality', [0, .25, .5, 1])
def test_fixed_scale_and_endpoints(quality):
    score = score_result(result(quality))
    assert score['overall_score'] == pytest.approx(quality * 100)
    assert sum(score['overall_components'].values()) == score['overall_score']


def test_known_mixed_components():
    row = result(0)
    row.update(topic_micro_f1=.9, type_accuracy=.8, exact_match=.4,
               score_mae={'etki': .3, 'ciddiyet': .6, 'aciliyet': 0})
    assert score_result(row)['overall_score'] == pytest.approx(78)


@pytest.mark.parametrize('metric,value', [('topic_micro_f1', 1.1), ('exact_match', float('nan')),
                                        ('type_accuracy', -.1)])
def test_invalid_metrics_are_rejected(metric, value):
    row = result(.5)
    row[metric] = value
    with pytest.raises(ValueError):
        score_result(row)


def test_invalid_prediction_is_penalized_without_a_second_validity_multiplier():
    truth = {'etiketler': ['EMEK'], 'tur': 'SOMUT_GELISME', 'etki': 1, 'ciddiyet': 1, 'aciliyet': 1}
    metrics = summarize([{'truth': truth, 'prediction': truth, 'seconds': .1},
                         {'truth': truth, 'error': 'invalid', 'seconds': .1}])
    # F1=2/3; type=exact=.5; normalized score accuracy=.5.
    assert score_result(metrics)['overall_score'] == pytest.approx(55)


def test_rank_upgrade_is_pool_independent_and_missing_models_stay_missing():
    row = result(.5)
    report = {'results': [copy.deepcopy(row)], 'pairs': [], 'statuses': [{'id': 'missing'}]}
    score_report(report)
    old_score = report['results'][0]['overall_score']
    better = {**result(1), 'model_id': 'two'}
    report['results'].append(better)
    score_report(report)
    assert [r['model_id'] for r in report['results']] == ['two', 'one']
    assert report['results'][1]['overall_score'] == old_score
    assert len(report['results']) == 2
