#!/usr/bin/env python3
"""Verify each GGUF pair in the Android native engine and publish immutable URLs."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    from huggingface_hub import HfApi, hf_hub_download
    import shutil
    api = HfApi()
    assert api.whoami()['name'] == 'efebaykara'
    mobile = ROOT / 'workspace/mobile'
    published = mobile / 'published'
    published.mkdir(exist_ok=True)
    index = read(ROOT / 'workspace/releases/models-v1.0.0/release-index.json')['models']
    app = ROOT.parent.parent
    order = ['qwen3.5-0.8b-base', 'qwen3.5-2b-base', 'llama-3.2-1b', 'llama-3.2-3b',
             'octen-webgr-0.6b', 'spark-x2.6-1.7b', 'limite-1b-violetto', 'lfm3-base', 'lfm3-posttrained']
    for entry in sorted(index, key=lambda e: order.index(e['model_id']) if e['model_id'] in order else 99):
        model_id, repo = entry['model_id'], entry['repo_id']
        if model_id == 'gpt-x3-150m':
            continue  # GPT's GDN2 pair is exported/published separately as ONNX.
        if (published / f'{model_id}.json').exists():
            continue
        for attempt in range(240):
            paths = [mobile / f'{model_id}-{v}-Q4_K_M.json' for v in ('original', 'distilled')]
            if all(p.exists() for p in paths):
                break
            time.sleep(5)
        else:
            print('MISSING EXPORT', model_id, flush=True)
            continue
        states = [read(p) for p in paths]
        try:
            for state in states:
                log = mobile / f'{model_id}-{state["variant"]}-native.log'
                with log.open('w') as stream:
                    subprocess.run([str(app / 'build/host-smoke/gundem-smoke'), str(mobile / state['file']),
                        str(app / 'app/src/main/assets/prompts')], stdout=stream, stderr=subprocess.STDOUT,
                        check=True, timeout=240)
                text = log.read_text(errors='replace')
                if 'CLASSIFY\n' not in text:
                    raise ValueError('Native generation did not start')
                state['native_classification_valid'] = 'CLASSIFY\nOK\n' in text
                print('NATIVE OK', model_id, state['variant'], flush=True)
            staging = mobile / 'upload' / model_id
            (staging / 'gguf').mkdir(parents=True, exist_ok=True)
            for state in states:
                source, target = mobile / state['file'], staging / 'gguf' / state['file']
                if not target.exists():
                    import os
                    os.link(source, target)
            write(staging / 'mobile-catalog.json', states)
            card = Path(hf_hub_download(repo, 'README.md')).read_text().split('\n## Mobile GGUF pair')[0]
            card += '\n## Mobile GGUF pair\n\nOriginal and fine-tuned Q4_K_M files, tested with the Gündem native engine.\n\n'
            card += '| Variant | GGUF | Gündem Score v1 (HF/BF16) |\n| --- | --- | ---: |\n'
            for state in states:
                card += f'| {state["variant"]} | [{state["file"]}](gguf/{state["file"]}) | {state["evaluation"]["overall_score"]:.1f} |\n'
            card += '\nThese scores come from the HF/BF16 experiment; mobile GGUF quality was not rescored. Both files retain the base model license and notices included in this repository. The original model is not fine-tuned.\n'
            (staging / 'README.md').write_text(card)
            result = api.upload_folder(repo_id=repo, folder_path=staging,
                commit_message='Publish native-tested original and fine-tuned mobile GGUF pair')
            api.create_tag(repo, tag='mobile-v1.1.0', revision=result.oid, exist_ok=True)
            info = api.model_info(repo, revision=result.oid, files_metadata=True)
            for state in states:
                remote = next(s for s in info.siblings if s.rfilename == 'gguf/' + state['file'])
                assert remote.size == state['size'] and remote.lfs.sha256 == state['sha256']
                state['url'] = f'https://huggingface.co/{repo}/resolve/{result.oid}/gguf/{state["file"]}'
            write(published / f'{model_id}.json', states)
            print('PUBLISHED PAIR', model_id, result.oid, flush=True)
        except Exception as error:
            write(mobile / f'{model_id}-publish-failed.json', {'error': f'{type(error).__name__}: {error}'})
            print('FAILED', model_id, str(error), flush=True)


if __name__ == '__main__':
    main()
