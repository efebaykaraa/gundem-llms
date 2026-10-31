#!/usr/bin/env python3
"""Prepare portable adapter repositories and GitHub assets; never uploads files."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(source, target)
    if digest(source) != digest(target):
        raise ValueError(f'Existing artifact differs: {target}')


def prepare(namespace, out):
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    from lab.modeling import template_hash
    api = HfApi()
    summary = read(ROOT / 'workspace/reports/leaderboard.json')
    if summary['completed_variants'] != summary['expected_variants']:
        raise ValueError('Complete the evaluation suite before preparing a release')
    results = {r['model_id']: r for r in summary['results'] if r['variant'] == 'distilled'}
    assets, models = out / 'github-assets', out / 'models'
    assets.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    # Extract only the exact stdlib validation helpers, without collection/API code.
    source = (ROOT / 'pipeline.py').read_text()
    tree = ast.parse(source)
    validation = '\n\n'.join(ast.get_source_segment(source, node) for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in ('escape_title', 'validate_label'))
        or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
             t.id in ('TOPICS', 'TYPES', 'FIELDS') for t in node.targets))) + '\n'
    entries = []
    for model_id, result in results.items():
        directory = ROOT / 'workspace/experiments' / model_id
        spec = read(directory / 'model.lock.json')
        adapter = directory / 'adapter'
        repo_name = spec['repo'].split('/')[-1] + '-Gundem-LoRA'
        repo_id = namespace + '/' + repo_name
        target = models / repo_name
        target.mkdir(exist_ok=True)
        info = api.model_info(spec['repo'], revision=spec['revision'])
        license_id = getattr(info.card_data, 'license', None)
        if license_id not in ('apache-2.0', 'llama3.2', 'cc-by-nc-4.0', 'other'):
            raise ValueError(f'Unreviewed model license: {spec["repo"]}: {license_id}')
        files = [s.rfilename for s in info.siblings if '/' not in s.rfilename and
                 any(term in s.rfilename.lower() for term in ('license', 'notice', 'use_policy'))]
        for name in files:
            shutil.copy2(hf_hub_download(spec['repo'], name, revision=spec['revision']), target / name)
        if not any(name.lower().startswith('license') for name in files):
            if license_id != 'apache-2.0':
                raise ValueError(f'License text absent: {spec["repo"]}')
            with urllib.request.urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt', timeout=30) as response:
                (target / 'LICENSE').write_bytes(response.read())
        shutil.copy2(ROOT.parent.parent / 'LICENSE', target / 'LICENSE-CODE-GUNDEM')
        attribution = f'Gündem response-distillation LoRA adaptation of {spec["repo"]}, revision {spec["revision"]}.\n'
        attribution += 'Modified artifact: adapter weights trained on DeepSeek-labelled Turkish headlines.\n'
        attribution += 'Model and tokenizer licensing follows the upstream license files. Gündem helper code: MIT.\n'
        if license_id == 'llama3.2':
            attribution += 'Built with Llama\nLlama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.\n'
        notice = target / 'NOTICE'
        existing = notice.read_text() if notice.exists() else ''
        notice.write_text(existing + '\n' + attribution)
        link(adapter / 'adapter_model.safetensors', target / 'adapter_model.safetensors')
        config = read(adapter / 'adapter_config.json')
        config.update(base_model_name_or_path=spec['repo'], revision=spec['revision'])
        write(target / 'adapter_config.json', config)
        for pattern in ('tokenizer*', 'chat_template*', 'special_tokens_map.json', 'added_tokens.json',
                        'vocab.*', 'merges.txt', '*.model'):
            for path in adapter.glob(pattern):
                if path.is_file():
                    shutil.copy2(path, target / path.name)
        tokenizer = AutoTokenizer.from_pretrained(target, trust_remote_code=True)
        if template_hash(tokenizer) != result['chat_template_sha256']:
            raise ValueError(f'Tokenizer template changed: {model_id}')
        shutil.copy2(ROOT / 'student-system.txt', target / 'student-system.txt')
        shutil.copy2(ROOT / 'lab/modeling.py', target / 'gundem_modeling.py')
        shutil.copy2(ROOT / 'release_infer.py', target / 'release_infer.py')
        (target / 'gundem_validation.py').write_text(validation)
        requirements = ''.join((ROOT / name).read_text() + '\n' for name in
                               ('requirements-train.txt', 'requirements-kernels.txt'))
        (target / 'requirements.txt').write_text(requirements)
        write(target / 'evaluation.json', result)
        write(target / 'release-manifest.json', {'release': 'models-v1.0.0', 'repo_id': repo_id,
              'model': spec, 'training': read(adapter / 'distill-manifest.json'), 'evaluation': result,
              'adapter_sha256': digest(target / 'adapter_model.safetensors'), 'license': license_id,
              'scoring': summary['scoring']})
        extra = ''
        if license_id == 'other':
            extra += '\nLFM Open License v1.0 applies, including its commercial-use conditions. See LICENSE.\n'
        if license_id == 'llama3.2':
            extra += '\n**Built with Llama.** Base-model access requires accepting Meta’s terms on Hugging Face. See LICENSE.txt and USE_POLICY.md.\n'
        if license_id == 'cc-by-nc-4.0':
            extra += '\n**Non-commercial (CC BY-NC 4.0).** Octen is a specialized URL retriever; this release tests transfer to news JSON, not retrieval quality. See LICENSE and NOTICE.\n'
        if model_id.startswith('qwen3.5'):
            extra += '\nText-only adaptation of Qwen3.5. The supplied loader uses Qwen3_5ForCausalLM; vision components were not trained.\n'
        if model_id == 'limite-1b-violetto':
            extra += '\nExperimental adaptation: explicit task ChatML replaces the fixed mathematical prompt. Direct-weight LoRA paths bypass feature dropout during training.\n'
        if model_id == 'spark-x2.6-1.7b':
            extra += '\nUse the supplied loader for compatibility between the pinned remote code and Transformers version.\n'
        license_yaml = f'license: {license_id}\n'
        if license_id == 'other':
            license_yaml += 'license_name: lfm1.0\nlicense_link: LICENSE\n'
        score = result['overall_score']
        card = f'''---
language:
- tr
{license_yaml}base_model: {spec['repo']}
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- response-distillation
- turkish-news
- headline-classification
---

# {repo_name}

LoRA adapter for **Turkish news headline classification**. Requires the pinned base model; this is not a standalone full checkpoint or a general chat assistant.

Base: [{spec['repo']}](https://huggingface.co/{spec['repo']}) · revision `{spec['revision']}`.

DeepSeek-generated answers were used for assistant-only supervised fine-tuning (response distillation). No DeepSeek weights or logits were used; the student's parameter count is unchanged. The split contains 825 training, 107 validation and 101 test examples. Three epochs, lr=0.0002, LoRA r16/alpha32, accumulation=8, max_length=512, seed=42.

## Use

Download this repository, install a suitable CUDA PyTorch build and the supplied requirements, then run from the downloaded folder:

```bash
python -m pip install -r requirements.txt
python release_infer.py --model . --headline "İşçiler ücret artışı için greve başladı"
```

The supplied loader handles architecture compatibility, the exact task template, pinned base revision, CPU fp32 adapter merging and CUDA BF16 inference. Allow CPU RAM for the full fp32 base model and GPU memory for BF16 inference. Loading the base model executes its remote model code. Review it before use. The result contains `prediction` or an explicit `error`; generation is not constrained by a JSON grammar.

The model receives only the headline and `student-system.txt`. Output fields: `etiketler` (topics), `tur` (event type), and `etki`, `ciddiyet`, `aciliyet` (integer scores 0–3).

## Evaluation

| Metric | Value |
| --- | ---: |
| Gündem Score v1 | {score:.1f} / 100 |
| All-field exact match | {result['exact_match']:.1%} |
| Topic micro-F1 | {result['topic_micro_f1']:.1%} |
| Type accuracy | {result['type_accuracy']:.1%} |
| Invalid outputs | {result['invalid']} / {result['count']} |

HF/CUDA BF16, greedy, batch=4, maximum 160 new tokens, no JSON grammar. Gündem Score = 30×topic F1 + 20×type accuracy + 30×(1−mean score MAE/3) + 20×exact match (fractional metrics). It measures this task's teacher-label agreement, not general LLM intelligence or an accuracy percentage. Small differences on this previously evaluated, single-seed test do not establish superiority. Batch size and numerical runtime changes can affect outputs. See `evaluation.json` and `release-manifest.json`.

## Licensing and limitations

{attribution}
{extra}
The task uses a small, domain-specific dataset. Inspect predictions before relying on them. Raw headlines, teacher answers and the full dataset are not redistributed. Model licensing is separate from the MIT-licensed Gündem helper code.

[Training and benchmark tools](https://github.com/efebaykaraa/Gundem) · [All released models](https://github.com/efebaykaraa/Gundem/releases/tag/models-v1.0.0)
'''
        (target / 'README.md').write_text(card)
        # ZIP is adapter-only. GGUF binaries are separate GitHub assets.
        archive = assets / (repo_name + '.zip')
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            for path in sorted(target.iterdir()):
                if path.is_file():
                    bundle.write(path, arcname=repo_name + '/' + path.name)
        entries.append({'model_id': model_id, 'repo_id': repo_id, 'folder': str(target.relative_to(out)),
                        'adapter_asset': archive.name, 'license': license_id, 'overall_score': score,
                        'gguf': None})
        gguf_source = {'lfm3-posttrained': ROOT / 'data-deepseek/gguf/gundem-lfm-distilled-Q4_K_M.gguf',
                       'lfm3-base': ROOT / 'data-deepseek-base/gguf/gundem-lfm-base-distilled-Q4_K_M.gguf'}.get(model_id)
        if gguf_source and gguf_source.is_file():
            gguf_name = repo_name.removesuffix('-LoRA') + '-Q4_K_M.gguf'
            link(gguf_source, assets / gguf_name)
            link(gguf_source, target / 'gguf' / gguf_name)
            entries[-1]['gguf'] = gguf_name
            with (target / 'README.md').open('a') as stream:
                stream.write(f'\n## GGUF\n\n`gguf/{gguf_name}` is a standalone Q4_K_M model for compatible llama.cpp/LM Studio runtimes. The metrics above evaluate the HF adapter in BF16, not this quantized artifact.\n')
        print('Prepared:', repo_id, flush=True)
    write(out / 'release-index.json', {'tag': 'models-v1.0.0', 'models': entries})
    write(assets / 'release-index.json', {'tag': 'models-v1.0.0', 'models': entries})
    for name in ('leaderboard.json', 'leaderboard.csv', 'overall-tr.png', 'overall-en.png'):
        shutil.copy2(ROOT / 'workspace/reports' / name, assets / name)
    with zipfile.ZipFile(assets / 'LICENSES.zip', 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for entry in entries:
            folder = out / entry['folder']
            for path in sorted(folder.iterdir()):
                if path.is_file() and any(term in path.name.lower() for term in ('license', 'notice', 'use_policy')):
                    bundle.write(path, arcname=folder.name + '/' + path.name)
    checksum = '\n'.join(f'{digest(path)}  {path.name}' for path in sorted(assets.iterdir())
                         if path.is_file() and path.name != 'SHA256SUMS.txt')
    (assets / 'SHA256SUMS.txt').write_text(checksum + '\n')
    notes = ['# Gündem models · v1.0.0', '',
             'Ten response-distilled LoRA adapters for Turkish news headline classification, plus standalone Q4_K_M GGUFs for the two LFM variants.', '',
             'ZIP assets include adapter weights, tokenizer, pinned base revision, model card, licensing, evaluation and a CUDA inference script. Base weights are downloaded separately. GGUF assets are standalone.', '',
             '| Model | HF repository | Gündem Score | License |', '| --- | --- | ---: | --- |']
    for entry in entries:
        notes.append(f'| {entry["model_id"]} | [{entry["repo_id"]}](https://huggingface.co/{entry["repo_id"]}) | {entry["overall_score"]:.1f} | {entry["license"]} |')
    notes += ['', 'Scores measure teacher-label agreement on 101 headlines: HF/CUDA BF16, greedy, batch=4, no JSON grammar. They are not general LLM capability or phone performance. Close scores are not evidence of meaningful superiority. GGUF artifacts were quantized separately; these scores do not describe GGUF inference.', '',
              'Use the README inside each ZIP. Llama: Built with Llama; base access and Meta terms apply. Octen: non-commercial CC BY-NC 4.0. LFM: LFM Open License v1.0. Other models: upstream Apache-2.0. License files and attribution notices are included per model.', '',
              'Download LICENSES.zip alongside standalone GGUF files for the applicable licenses and attribution notices.', '',
              'Verify downloads with `sha256sum -c SHA256SUMS.txt`. Raw data, API keys, optimizer state and training environments are not included.']
    (out / 'release-notes.md').write_text('\n'.join(notes) + '\n')
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'workspace/releases/models-v1.0.0')
    args = parser.parse_args()
    prepare(args.namespace, args.out.resolve())


if __name__ == '__main__':
    main()
