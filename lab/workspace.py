"""Central workspace, with immutable data snapshots and links to legacy runs."""
from pathlib import Path
import os
import shutil
from .common import ROOT, sha, validate_dataset


def link(source, target):
    source, target = Path(source).resolve(), Path(target)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() != source:
            raise ValueError(f'Conflicting workspace link: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())


def initialize(config, source=None):
    work = ROOT / config['workspace']
    destination = ROOT / config['dataset']
    if source:
        source = Path(source).resolve()
        validate_dataset(source)
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('manifest.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl'):
            target = destination / name
            if target.exists():
                if sha(source / name) != sha(target):
                    raise ValueError(f'Different dataset already exists: {target}')
            else:
                shutil.copy2(source / name, target)
    validate_dataset(destination)
    for item in config['models']:
        if item.get('legacy_adapter'):
            original = ROOT / item['legacy_adapter']
            if original.is_dir():
                link(original, work / 'experiments' / item['id'] / 'adapter')
    for name, source in [('training', ROOT / 'data-deepseek/venv-train'),
                         ('converter', ROOT / '.venv-convert-lms')]:
        if source.is_dir():
            link(source, work / 'envs' / name)
    (work / 'reports').mkdir(parents=True, exist_ok=True)
    (work / 'README.md').write_text(
        '# Local workspace\n\n- `datasets/news-v1/`: frozen shared split files.\n'
        '- `experiments/<model>/`: revision lock, training logs, adapter, original/distilled evaluations.\n'
        '- `reports/`: automatically generated tables and bilingual charts.\n'
        '- `envs/`: links to existing environments.\n\n'
        'Legacy LFM adapters are linked, not moved or duplicated. This directory is excluded from Git.\n',
        encoding='utf-8')
    return destination
