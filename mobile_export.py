#!/usr/bin/env python3
"""Export the measured original/finetuned pairs to real Q4_K_M mobile artifacts."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--out', type=Path, default=ROOT / 'workspace/mobile')
    args = parser.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from peft import PeftModel
    from lab.modeling import load_model, configure_tokenizer
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = json.loads((ROOT / 'workspace/reports/leaderboard.json').read_text())['results']
    # Each pair is exported once, using exactly its measured base revision and adapter.
    ids = list(dict.fromkeys(r['model_id'] for r in results))
    if args.models:
        ids = args.models
    converter = ROOT / '.venv-convert-lms/bin/python'
    llama = ROOT.parent.parent / 'vendor/llama.cpp'
    for model_id in ids:
        if model_id == 'gpt-x3-150m':
            print('GPT-X3 uses export_gpt_mobile.py (ONNX); skipping GGUF', flush=True)
            continue
        directory = ROOT / 'workspace/experiments' / model_id
        spec = json.loads((directory / 'model.lock.json').read_text())
        for variant in ('original', 'distilled'):
            target = output / f'{model_id}-{variant}-Q4_K_M.gguf'
            metadata = target.with_suffix('.json')
            report = next(r for r in results if r['model_id'] == model_id and r['variant'] == variant)
            state = {'model_id': model_id, 'variant': variant, 'model': spec, 'evaluation': report,
                     'adapter_sha256': digest(directory / 'adapter/adapter_model.safetensors') if variant == 'distilled' else None}
            if target.exists() and metadata.exists():
                old = json.loads(metadata.read_text())
                if old['model'] != spec or old['adapter_sha256'] != state['adapter_sha256'] or old['sha256'] != digest(target):
                    raise ValueError(f'Incompatible existing artifact: {target}')
                print('REUSED', target.name, flush=True)
                continue
            legacy = {('lfm3-posttrained', 'original'): ROOT / 'data-deepseek/gguf/gundem-lfm-original-Q4_K_M.gguf',
                      ('lfm3-posttrained', 'distilled'): ROOT / 'data-deepseek/gguf/gundem-lfm-distilled-Q4_K_M.gguf',
                      ('lfm3-base', 'distilled'): ROOT / 'data-deepseek-base/gguf/gundem-lfm-base-distilled-Q4_K_M.gguf'}.get((model_id, variant))
            try:
                print('EXPORTING', model_id, variant, flush=True)
                if legacy and legacy.is_file():
                    if not target.exists():
                        os.link(legacy, target)
                else:
                    with tempfile.TemporaryDirectory(prefix='convert-', dir=output) as scratch:
                        scratch = Path(scratch)
                        base_path = snapshot_download(spec['repo'], revision=spec['revision'], local_files_only=True,
                            allow_patterns=['*.json', '*.safetensors', '*.py', '*.txt', '*.model', '*.tiktoken', '*.jinja', '*.jinja2'])
                        model = load_model(base_path, torch.float32, revision=spec['revision'])
                        model.train()
                        if variant == 'distilled':
                            model = PeftModel.from_pretrained(model, directory / 'adapter').merge_and_unload(safe_merge=True)
                            model.train()
                        model.save_pretrained(scratch, safe_serialization=True)
                        tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(base_path, trust_remote_code=True), spec['prompt_style'])
                        tokenizer.save_pretrained(scratch)
                        del model, tokenizer
                        gc.collect()
                        with (output / f'{model_id}-{variant}.log').open('w') as log:
                            subprocess.run([str(converter), str(ROOT / 'export_gguf.py'), '--model', str(scratch),
                                '--llama-dir', str(llama), '--quantizer', str(llama / 'build-distill/bin/llama-quantize'),
                                '--out', str(target)], stdout=log, stderr=subprocess.STDOUT, check=True)
                state.update(status='exported', file=target.name, size=target.stat().st_size, sha256=digest(target))
                write(metadata, state)
                (output / f'{model_id}-{variant}-failed.json').unlink(missing_ok=True)
                print('EXPORTED', target.name, flush=True)
            except Exception as error:
                state.update(status='failed', error=f'{type(error).__name__}: {error}')
                write(output / f'{model_id}-{variant}-failed.json', state)
                print('FAILED', model_id, variant, state['error'], flush=True)


if __name__ == '__main__':
    main()
