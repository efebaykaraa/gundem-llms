from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.runner import invalidate_distilled_evaluation
from lab import runner
import pytest


def test_retraining_archives_stale_distilled_results_but_keeps_original(tmp_path):
    for name in ('original.json', 'distilled.json', 'distilled.json.predictions.jsonl', 'distilled.log'):
        (tmp_path / name).write_text(name)
    archive = invalidate_distilled_evaluation(tmp_path)
    assert (tmp_path / 'original.json').read_text() == 'original.json'
    assert not (tmp_path / 'distilled.json').exists()
    assert (archive / 'distilled.json').read_text() == 'distilled.json'
    assert (archive / 'distilled.json.predictions.jsonl').exists()
    assert invalidate_distilled_evaluation(tmp_path) is None


def test_changed_evaluation_identity_is_excluded_from_aggregation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'validate_dataset', lambda _: ({}, None))
    monkeypatch.setattr(runner, 'fingerprint', lambda *args: 'expected')
    def complete_with_changed_identity(args, log):
        runner.write_json(tmp_path / 'original.json', {'fingerprint': 'changed'})
    monkeypatch.setattr(runner, 'command', complete_with_changed_identity)
    with pytest.raises(ValueError, match='identity changed'):
        runner.evaluate_one({'id': 'test'}, 'original', {'evaluation': {}},
                            tmp_path, tmp_path, tmp_path / 'plan.json')
    assert not (tmp_path / 'original.json').exists()
    assert len(list(tmp_path.glob('original-rejected-*.json'))) == 1
