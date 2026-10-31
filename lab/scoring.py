"""Fixed, task-specific quality score; independent of hardware and competitor pool."""
import math

SCORING = {
    'id': 'gundem-score-v1',
    'name': 'Gündem Score',
    'range': [0, 100],
    'weights': {'topic_micro_f1': 30, 'type_accuracy': 20,
                'score_accuracy': 30, 'exact_match': 20},
    'score_accuracy': '1 - mean(etki_mae, ciddiyet_mae, aciliyet_mae) / 3',
    'invalid_policy': 'Already penalized in the input metrics; no extra validity multiplier.',
    'scope': 'Teacher-label agreement for this headline task; not general LLM capability or accuracy percent.',
}


def bounded(value, maximum, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be numeric')
    if not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError(f'{name} must be finite and in [0, {maximum}]')
    return value


def score_result(result):
    """Return unrounded weighted points; invalid predictions are already penalized."""
    mean_mae = sum(bounded(result['score_mae'][field], 3, field + '_mae')
                   for field in ('etki', 'ciddiyet', 'aciliyet')) / 3
    normalized = {key: bounded(result[key], 1, key)
                  for key in ('topic_micro_f1', 'type_accuracy', 'exact_match')}
    normalized['score_accuracy'] = 1 - mean_mae / 3
    components = {key: weight * normalized[key] for key, weight in SCORING['weights'].items()}
    return {'overall_score': sum(components.values()), 'overall_components': components,
            'overall_score_version': SCORING['id']}


def score_report(report):
    """Upgrade legacy summary JSON too, without weights, predictions or inference."""
    report['scoring'] = SCORING
    for result in report['results']:
        result.update(score_result(result))
    report['results'].sort(key=lambda r: (-r['overall_score'], -r['exact_match'],
                                        -r['topic_micro_f1'], r['model_id'], r['variant']))
    for pair in report.get('pairs', []):
        variants = {r['variant']: r for r in report['results'] if r['model_id'] == pair['id']}
        if 'original' in variants and 'distilled' in variants:
            pair['delta_distilled_minus_original']['overall_score'] = (
                variants['distilled']['overall_score'] - variants['original']['overall_score'])
    return report
