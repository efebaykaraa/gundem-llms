"""Dataset integrity, atomic state and experiment identity (standard library only)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(directory):
    import pipeline
    from train import check_splits
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    data = {s: pipeline.rows(directory / f'{s}.jsonl') for s in ('train', 'validation', 'test')}
    check_splits(data)
    for split, records in data.items():
        if not records or len(records) != manifest['counts'][split]:
            raise ValueError(f'Empty or mismatched {split} count')
        if sha(directory / f'{split}.jsonl') != manifest['split_sha256'][split]:
            raise ValueError(f'Dataset modified: {split}')
        for row in records:
            if pipeline.digest(row['messages'][0]['content']) != manifest['student_prompt_sha256']:
                raise ValueError('Student system prompt changed')
            if row['messages'][1]['content'] != pipeline.escape_title(row['title']):
                raise ValueError('Student user message must be headline only')
    return manifest, data


def validate_training(adapter, manifest, training, spec):
    from .loss import LOSS_CONTRACT
    trained = read_json(Path(adapter) / 'distill-manifest.json')
    for key, expected in [('dataset_manifest', manifest), ('hyperparameters', training),
                          ('student', spec['repo']), ('revision', spec['revision'])]:
        if trained.get(key) != expected:
            raise ValueError(f'Trained model differs from suite: {key}')
    if trained.get('prompt_style', 'native') != spec.get('prompt_style', 'auto'):
        raise ValueError('Training/evaluation prompt styles differ')
    if trained.get('loss_contract') != LOSS_CONTRACT:
        legacy = {'LiquidAI/LFM3-1.7B': 'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7',
                  'LiquidAI/LFM3-1.7B-Base': '545fc8f4c7b5fa9f405105274cd29c4f8f44b526'}
        if (trained.get('loss_contract') is not None or legacy.get(spec['repo']) != spec['revision']
                or trained.get('versions', {}).get('transformers') != '5.20.2'):
            raise ValueError('Adapter does not use the common normalized training objective')
    return trained
