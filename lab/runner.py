"""Resumable sequential original/distilled experiment orchestration."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .common import ROOT, read_json, write_json, validate_dataset, validate_training
from .evaluate import fingerprint
from .workspace import initialize
from .report import aggregate, plot


def configuration(path):
    config = read_json(path)
    ids = [m['id'] for m in config['models']]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[a-z0-9][a-z0-9._-]*', i) for i in ids):
        raise ValueError('Model IDs must be unique safe directory names')
    evaluation = config['evaluation']
    if evaluation['batch_size'] < 1 or evaluation['max_new_tokens'] < 1 or evaluation['warmup'] < 0:
        raise ValueError('Invalid evaluation batch/token/warmup setting')
    if evaluation['dtype'] != 'bfloat16' or evaluation['decoding'] != 'greedy' or evaluation['json_constraints']:
        raise ValueError('This suite implements bfloat16 greedy generation without JSON constraints')
    return config


def state(directory, status, detail='', **extra):
    value = {'status': status, 'detail': detail,
             'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}
    write_json(directory / 'state.json', value)
    print(f'[{directory.name}] {status}: {detail}', flush=True)


def lock_spec(spec, directory):
    from train import configure_hub_download
    configure_hub_download()
    from huggingface_hub import HfApi, hf_hub_download
    path = directory / 'model.lock.json'
    if path.exists():
        pinned = read_json(path)
        if any(pinned[k] != spec[k] for k in ('id', 'repo', 'prompt_style')):
            raise ValueError('Existing model lock differs; choose a new experiment directory')
        if spec['revision'] != 'main' and pinned['revision'] != spec['revision']:
            raise ValueError('Pinned model revision differs from config')
    else:
        info = HfApi().model_info(spec['repo'], revision=spec['revision'], timeout=30)
        pinned = dict(spec, revision=info.sha)
    # Confirm that gated weights/config access is granted; never request access on behalf of the user.
    hf_hub_download(pinned['repo'], 'config.json', revision=pinned['revision'])
    write_json(path, pinned)
    return pinned


def command(args, log):
    print('+ ' + ' '.join(str(a) for a in args), flush=True)
    with Path(log).open('a', encoding='utf-8') as stream:
        stream.write('\nCOMMAND: ' + ' '.join(str(a) for a in args) + '\n')
        stream.flush()
        subprocess.run([str(a) for a in args], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})


def refresh(config, work):
    report = aggregate(config, work, work / 'reports')
    try:
        plot(report, work / 'reports')
    except ImportError:
        print('Charts need Matplotlib: install requirements-report.txt in this environment.', flush=True)
    return report


def invalidate_distilled_evaluation(directory):
    files = [directory / name for name in ('distilled.json', 'distilled.json.predictions.jsonl',
                                           'distilled.json.partial.predictions.jsonl', 'distilled.log')]
    present = [path for path in files if path.exists()]
    if present:
        archive = directory / ('previous-evaluation-' + str(time.time_ns()))
        archive.mkdir()
        for path in present:
            path.rename(archive / path.name)
        return archive


def train_one(spec, config, directory, dataset):
    adapter = directory / 'adapter'
    manifest, _ = validate_dataset(dataset)
    if (adapter / 'distill-manifest.json').is_file():
        validate_training(adapter, manifest, config['training'], spec)
        print(f'[{spec["id"]}] existing compatible adapter reused', flush=True)
        return adapter
    invalidate_distilled_evaluation(directory)
    arguments = [sys.executable, ROOT / 'train.py', 'train', '--data', dataset, '--out', adapter,
                 '--student', spec['repo'], '--revision', spec['revision'],
                 '--prompt-style', spec['prompt_style'], '--progress', 'plain', '--uniform-loss']
    for key, value in config['training'].items():
        arguments += ['--' + key.replace('_', '-'), str(value)]
    if adapter.exists() and any(adapter.iterdir()):
        checkpoints = [p for p in adapter.glob('checkpoint-*') if (p / 'trainer_state.json').is_file()
                       and (p / 'optimizer.pt').is_file() and (p / 'adapter_model.safetensors').is_file()]
        if checkpoints:
            latest = max(checkpoints, key=lambda p: int(p.name.split('-')[-1]))
            arguments += ['--resume', latest]
        else:
            archive = directory / ('incomplete-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            adapter.rename(archive)
            print(f'Incomplete pre-checkpoint files preserved in {archive}', flush=True)
    state(directory, 'training')
    command(arguments, directory / 'train.log')
    validate_training(adapter, manifest, config['training'], spec)
    return adapter


def evaluate_one(spec, variant, config, directory, dataset, plan):
    adapter = directory / 'adapter' if variant == 'distilled' else None
    manifest, _ = validate_dataset(dataset)
    if adapter:
        validate_training(adapter, manifest, config['training'], spec)
    target = directory / f'{variant}.json'
    expected = fingerprint(spec, manifest, config['evaluation'], adapter)
    if target.exists() and read_json(target).get('fingerprint') == expected:
        print(f'[{spec["id"]}] {variant} evaluation already complete', flush=True)
        return
    if target.exists():
        target.rename(directory / f'{variant}-previous-{time.time_ns()}.json')
    state(directory, f'evaluating_{variant}')
    args = [sys.executable, '-m', 'lab.evaluate', '--plan', plan, '--spec', directory / 'model.lock.json',
            '--dataset', dataset, '--out', target]
    if adapter:
        args += ['--adapter', adapter]
    command(args, directory / f'{variant}.log')
    if read_json(target)['fingerprint'] != expected:
        target.rename(directory / f'{variant}-rejected-{time.time_ns()}.json')
        raise ValueError('Evaluation identity changed during execution')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['init', 'preflight', 'run', 'report', 'status'])
    p.add_argument('--config', type=Path, default=ROOT / 'configs/models.json')
    p.add_argument('--source', type=Path, help='Prepared dataset to snapshot on init')
    p.add_argument('--models', nargs='+', help='Model IDs to run (default: all)')
    p.add_argument('--workspace', type=Path, help='Use a separate run workspace')
    p.add_argument('--evaluate-only', action='store_true', help='Evaluate existing adapters without training')
    a = p.parse_args()
    os.chdir(ROOT)
    config = configuration(a.config)
    if a.workspace:
        config['workspace'] = str(a.workspace.resolve())
    work = ROOT / config['workspace']
    selected = [s for s in config['models'] if not a.models or s['id'] in a.models]
    if a.models and set(a.models) != {s['id'] for s in selected}:
        p.error('Unknown --models ID')
    work.mkdir(parents=True, exist_ok=True)
    if a.action == 'status':
        for spec in config['models']:
            path = work / 'experiments' / spec['id'] / 'state.json'
            print(spec['id'], json.dumps(read_json(path), ensure_ascii=False) if path.exists() else 'pending')
        return 0
    lock = (work / '.suite.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('Another suite process owns this workspace; use status to inspect it')
    if a.action == 'report':
        report = refresh(config, work)
        print(f'{report["completed_variants"]}/{report["expected_variants"]}: {work / "reports/leaderboard.md"}')
        return 0
    dataset = initialize(config, a.source)
    plan = work / 'plan.json'
    if plan.exists():
        old = read_json(plan)
        if old['training'] != config['training'] or old['evaluation'] != config['evaluation']:
            raise ValueError('Run settings changed; use another --workspace')
    write_json(plan, config)
    if a.action == 'init':
        print('Workspace initialized:', work)
        return 0
    failures = 0
    for configured in selected:
        directory = work / 'experiments' / configured['id']
        directory.mkdir(parents=True, exist_ok=True)
        try:
            spec = lock_spec(configured, directory)
        except Exception as error:
            failures += 1
            state(directory, 'blocked_access' if type(error).__name__ == 'GatedRepoError' else 'preflight_failed',
                  type(error).__name__ + ': ' + str(error).splitlines()[0])
            refresh(config, work)
            continue
        if a.action == 'preflight':
            state(directory, 'ready', spec['revision'])
            continue
        try:
            evaluate_one(spec, 'original', config, directory, dataset, plan)
        except Exception as error:
            failures += 1
            state(directory, 'original_failed', f'{type(error).__name__}; see original.log')
        refresh(config, work)
        try:
            if not a.evaluate_only:
                train_one(spec, config, directory, dataset)
            evaluate_one(spec, 'distilled', config, directory, dataset, plan)
            state(directory, 'complete' if (directory / 'original.json').exists() else 'distilled_only')
        except Exception as error:
            failures += 1
            state(directory, 'failed', f'{type(error).__name__}: {str(error).splitlines()[0][:220]}; see logs')
        refresh(config, work)
    report = refresh(config, work)
    print(f'Completed variants: {report["completed_variants"]}/{report["expected_variants"]}', flush=True)
    print('Report:', work / 'reports/leaderboard.md', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
