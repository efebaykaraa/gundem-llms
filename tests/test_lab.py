import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.modeling import configure_tokenizer, PLAIN_TEMPLATE, CHATML_TEMPLATE
from lab.evaluate import summarize
from lab.report import aggregate
from lab.common import write_json

LABEL = {'etiketler': ['EMEK'], 'tur': 'SOMUT_GELISME', 'etki': 1, 'ciddiyet': 2, 'aciliyet': 0}


def test_prompt_fallback_is_explicit_and_native_template_is_preserved():
    tok = SimpleNamespace(chat_template=None, pad_token_id=1, eos_token_id=2)
    with pytest.raises(ValueError, match='No native'):
        configure_tokenizer(tok, 'native')
    configure_tokenizer(tok, 'auto')
    assert tok.chat_template == PLAIN_TEMPLATE
    tok.chat_template = 'vendor template'
    assert configure_tokenizer(tok, 'auto').chat_template == 'vendor template'
    assert configure_tokenizer(tok, 'task-chatml').chat_template == CHATML_TEMPLATE


def test_invalid_response_is_penalized_in_every_denominator():
    predictions = [{'truth': LABEL, 'prediction': LABEL, 'seconds': .1},
                   {'truth': LABEL, 'error': 'unfinished', 'seconds': .2}]
    result = summarize(predictions)
    assert result['count'] == 2 and result['invalid'] == 1
    assert result['exact_match'] == .5 and result['type_accuracy'] == .5
    assert result['topic_micro_f1'] == pytest.approx(2 / 3)
    assert set(result['score_mae'].values()) == {1.5}


@pytest.fixture
def report_setup(tmp_path):
    config = {'models': [{'id': 'one', 'label': 'One'}, {'id': 'two', 'label': 'Two'}]}
    root = tmp_path / 'workspace'
    result = dict(model_id='one', label='One', model='test/one', variant='original',
                  revision='abc', test_sha256='same-test', chat_template_sha256='same-template',
                  count=2, settings={'batch_size': 4}, backend='same-runtime', invalid=0,
                  exact_match=.5, topic_micro_f1=.7, type_accuracy=.5,
                  score_mae={'etki': .2, 'ciddiyet': .3, 'aciliyet': .4},
                  latency_seconds={'mean': .1}, parameters=100)
    write_json(root / 'experiments/one/original.json', result)
    return config, root, result


def test_pending_and_failed_models_are_not_fabricated_zero_scores(report_setup):
    config, root, result = report_setup
    write_json(root / 'experiments/two/state.json', {'status': 'blocked_access', 'detail': 'Login required'})
    report = aggregate(config, root, root / 'reports')
    assert report['expected_variants'] == 4 and report['completed_variants'] == 1
    assert len(report['results']) == 1 and report['pairs'] == []
    assert report['statuses'][1]['status'] == 'blocked_access'


@pytest.mark.parametrize('change', [{'test_sha256': 'different'}, {'backend': 'other'},
                                   {'settings': {'batch_size': 1}}])
def test_report_refuses_mixed_test_or_runtime(report_setup, change):
    config, root, result = report_setup
    result = {**result, 'variant': 'distilled', **change}
    write_json(root / 'experiments/one/distilled.json', result)
    with pytest.raises(ValueError, match='Cannot mix'):
        aggregate(config, root, root / 'reports')


def test_pair_requires_matching_revision_and_actual_template(report_setup):
    config, root, result = report_setup
    result = {**result, 'variant': 'distilled', 'chat_template_sha256': 'different'}
    write_json(root / 'experiments/one/distilled.json', result)
    with pytest.raises(ValueError, match='template mismatch'):
        aggregate(config, root, root / 'reports')


def test_pair_delta_is_new_minus_original(report_setup):
    config, root, result = report_setup
    result = {**result, 'variant': 'distilled', 'exact_match': .75}
    write_json(root / 'experiments/one/distilled.json', result)
    report = aggregate(config, root, root / 'reports')
    assert report['pairs'][0]['delta_distilled_minus_original']['exact_match'] == .25
