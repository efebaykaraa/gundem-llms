#!/usr/bin/env python3
"""Prepare Q4_K_M models and benchmark them sequentially through LM Studio."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pipeline

HERE = Path(__file__).resolve().parent


def run(*command, capture=False):
    command = [str(x) for x in command]
    print('+ ' + ' '.join(command), flush=True)
    return subprocess.run(command, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def prepare(args, original):
    pairs = [(original, args.old_gguf), (args.new_hf, args.new_gguf)]
    if all(out.is_file() for _, out in pairs):
        return
    for model, out in pairs:
        if not out.is_file() and not (model / 'config.json').is_file():
            raise ValueError(f'HF model not found: {model}; use --old-hf / --new-hf')
    llama = HERE.parent.parent / 'vendor/llama.cpp'
    build = llama / 'build-distill'
    quantizer = build / 'bin/llama-quantize'
    if not quantizer.is_file():
        run('cmake', '-S', llama, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
            '-DGGML_NATIVE=OFF', '-DGGML_CPU_KLEIDIAI=OFF', '-DGGML_CPU_REPACK=OFF',
            '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_CURL=OFF')
        run('cmake', '--build', build, '--target', 'llama-quantize', '-j', args.jobs)
    env = HERE / '.venv-convert-lms'
    python = env / 'bin/python'
    uv = shutil.which('uv')
    if not python.is_file():
        if uv:
            run(uv, 'venv', '--python', sys.executable, env)
        else:
            run(sys.executable, '-m', 'venv', env)
    requirements = llama / 'requirements/requirements-convert_hf_to_gguf.txt'
    legacy = llama / 'requirements/requirements-convert_legacy_llama.txt'
    signature = hashlib.sha256(requirements.read_bytes() + legacy.read_bytes() + b'lfm3-transformers-5.20.2').hexdigest()
    marker = env / 'converter-ready'
    if not marker.is_file() or marker.read_text() != signature:
        installer = [uv, 'pip', 'install', '--python', python] if uv else [python, '-m', 'pip', 'install']
        run(*installer, '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.11.0')
        run(*installer, '-r', requirements)
        # LFM3's tokenizer declares TokenizersBackend, introduced in Transformers 5.
        run(*installer, 'transformers==5.20.2')
        marker.write_text(signature)
    for model, out in pairs:
        if not out.is_file():
            run(python, HERE / 'export_gguf.py', '--model', model, '--llama-dir', llama,
                '--quantizer', quantizer, '--out', out)


def main():
    os.chdir(HERE)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'data-deepseek/prepared')
    parser.add_argument('--new-manifest', type=Path, default=HERE / 'data-deepseek/merged/distill-manifest.json')
    parser.add_argument('--old-hf', type=Path)
    parser.add_argument('--new-hf', type=Path, default=HERE / 'data-deepseek/merged')
    parser.add_argument('--old-gguf', type=Path, default=HERE / 'data-deepseek/gguf/gundem-lfm-original-Q4_K_M.gguf')
    parser.add_argument('--new-gguf', type=Path, default=HERE / 'data-deepseek/gguf/gundem-lfm-distilled-Q4_K_M.gguf')
    parser.add_argument('--out', type=Path, default=HERE / 'data-deepseek/benchmark-lms.json')
    parser.add_argument('--port', type=int, default=1234)
    parser.add_argument('--gpu', default='max')
    parser.add_argument('--context-length', type=int, default=2048)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-only', action='store_true', help='Require existing GGUFs; skip converter installation')
    parser.add_argument('--comparison', choices=['original-distilled', 'starting-checkpoint'],
                        default='original-distilled', help='Use starting-checkpoint for post-trained+LoRA vs base+LoRA')
    args = parser.parse_args()
    if args.jobs < 1 or args.warmup < 0 or args.context_length < 512:
        parser.error('jobs >= 1, warmup >= 0 and context-length >= 512 required')
    lock_handle = (HERE / '.benchmark-lms.lock').open('a')
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another LM Studio benchmark/preparation is already running')
    lms = shutil.which('lms') or str(Path.home() / '.lmstudio/bin/lms')
    if not Path(lms).is_file():
        parser.error('lms not found; install LM Studio CLI')
    trained = json.loads(args.new_manifest.read_text())
    cache = Path(os.environ.get('HF_HUB_CACHE', str(Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))) / 'hub')))
    original = args.old_hf or cache / ('models--' + trained['student'].replace('/', '--')) / 'snapshots' / trained['revision']
    if not args.run_only:
        prepare(args, original)
    for path in (args.old_gguf, args.new_gguf):
        with path.open('rb') as stream:
            if stream.read(4) != b'GGUF':
                raise ValueError(f'Invalid GGUF: {path}')
    if args.prepare_only:
        print('GGUF modelleri hazır.', flush=True)
        return
    status = json.loads(run(lms, 'server', 'status', '--json', capture=True))
    if status['running']:
        args.port = status['port']
    else:
        run(lms, 'server', 'start', '--port', args.port, '--bind', '127.0.0.1')
    count = len(pipeline.rows(args.data / 'test.jsonl'))
    configs = []
    names = ('posttrained-lora', 'base-lora') if args.comparison == 'starting-checkpoint' else ('original', 'distilled')
    for name, path in zip(names, (args.old_gguf, args.new_gguf)):
        repo = f'Gundem/LFM3-{name}'
        key = f'{repo}/{path.name}'
        installed = json.loads(run(lms, 'ls', '--json', capture=True))
        if not any(key in (item.get('path'), item.get('modelKey')) for item in installed):
            run(lms, 'import', path.resolve(), '--symbolic-link', '--user-repo', repo, '--yes')
        for attempt in range(30):
            installed = json.loads(run(lms, 'ls', '--json', capture=True))
            found = [item for item in installed if item.get('path') == key]
            if found:
                break
            time.sleep(1)
        if not found:
            raise ValueError(f'LM Studio did not index the imported model: {key}')
        configs.append(dict(model=f'gundem-benchmark-{name}', model_key=found[0]['modelKey'],
                            base_url=f'http://127.0.0.1:{args.port}/v1', api_key_env='STUDENT_API_KEY',
                            max_tokens=160, temperature=0, response_format='json_schema', timeout=180,
                            extra_body={'reasoning_effort': 'none',
                                        'chat_template_kwargs': {'enable_thinking': False}}))

    class LmsClient(pipeline.ChatClient):
        def __init__(self, config):
            super().__init__(config)
            self.completed = 0
            self.loaded = False
            active = json.loads(run(lms, 'ps', '--json', capture=True))
            if any(item.get('identifier') == config['model'] for item in active):
                raise ValueError(f'Benchmark identifier already loaded: {config["model"]}')
            run(lms, 'load', config['model_key'], '--identifier', config['model'],
                '--gpu', args.gpu, '--context-length', args.context_length, '--parallel', '1', '--yes')
            self.loaded = True

        def classify(self, title, prompt):
            try:
                return super().classify(title, prompt)
            finally:
                self.completed += 1
                print(f'  istek {self.completed}/{count + args.warmup} (ısınma dahil)', flush=True)

        def warmup(self, title, prompt):
            try:
                self.classify(title, prompt)
            except ValueError:
                pass  # Invalid/truncated model output is measured on the test set.

        def close(self):
            if self.loaded:
                run(lms, 'unload', self.config['model'])

    args.backend = f'LM Studio Q4_K_M; gpu={args.gpu}; context={args.context_length}; JSON schema'
    pipeline.benchmark(args, configs=configs, client_factory=LmsClient)
    print(f'Rapor: {args.out.resolve()}', flush=True)
    try:
        run(sys.executable, HERE / 'plot_benchmark.py', '--input', args.out)
    except subprocess.CalledProcessError:
        print('Benchmark kaydedildi; grafik oluşturulamadı. plot_benchmark.py ile tekrar deneyin.',
              file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
