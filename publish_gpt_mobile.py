#!/usr/bin/env python3
"""Publish both standalone ONNX bundles after real Kotlin/JNI inference verification."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    from huggingface_hub import HfApi, hf_hub_download
    app = ROOT.parent.parent
    test = ET.parse(app / 'app/build/test-results/testDebugUnitTest/TEST-tr.gundem.GptOnnxRuntimeTest.xml').getroot()
    assert int(test.get('tests')) >= 2 and all(int(test.get(key, 0)) == 0 for key in ('failures', 'errors', 'skipped'))
    root = ROOT / 'workspace/mobile'
    source = root / 'gpt-onnx'
    staging = root / 'upload/gpt-x3-150m'
    (staging / 'onnx').mkdir(parents=True, exist_ok=True)
    states = []
    for variant in ('original', 'distilled'):
        path = source / f'gpt-x3-150m-{variant}-ONNX-INT8.zip'
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            manifest = json.loads(archive.read('bundle.json'))
        assert manifest['fp32_max_logit_error'] < .003
        shutil.copy2(path, staging / 'onnx' / path.name)
        states.append({'model_id': 'gpt-x3-150m', 'variant': variant, 'format': 'onnx', 'file': path.name,
            'size': path.stat().st_size, 'sha256': sha(path), 'evaluation': manifest['evaluation'],
            'fp32_max_logit_error': manifest['fp32_max_logit_error'], 'tokenizer_parity_cases': 106,
            'native_verification': 'Actual Kotlin/JNI/ONNX inference, cancellation and installation tested on host JVM'})
    api = HfApi()
    assert api.whoami()['name'] == 'efebaykara'
    repo = 'efebaykara/GPT-X3-150M-Gundem-LoRA'
    card = Path(hf_hub_download(repo, 'README.md')).read_text().split('\n## Mobile ONNX pair')[0]
    card += '\n## Mobile ONNX pair\n\nThe following **standalone full INT8 models** run in Gündem Android v0.9.1 or later. These ZIPs do not require downloading the base model or applying an adapter. They use ONNX Runtime for GDN2 inference; they are not GGUF models. `tokenizer.gguf` inside each ZIP contains only the exact BPE vocabulary, not inference weights.\n\n'
    card += '| Variant | Download | Gündem Score v1 (HF/BF16) |\n| --- | --- | ---: |\n'
    for state in states:
        card += f'| {state["variant"]} | [{state["file"]}](onnx/{state["file"]}) | {state["evaluation"]["overall_score"]:.1f} |\n'
    card += '\nThe original bundle uses the pinned, unadapted checkpoint. The distilled bundle merges the released LoRA into that same checkpoint. INT8 MatMul quantization retains FP32 embeddings and recurrent state. Tokenization matched HF on 106 Turkish/Unicode/whitespace cases. FP32 ONNX logits matched the pinned PyTorch model before quantization (see metadata). Actual production Kotlin/JNI code was exercised on the host JVM, including model installation, constrained JSON, cancellation and reset. This is not a phone-speed measurement. The scores above are from the HF/BF16 experiment, not a new ONNX quality evaluation.\n\nExport code adapted under Apache-2.0 from [ahmadw/GPT-X3-150M-ONNX](https://huggingface.co/ahmadw/GPT-X3-150M-ONNX/tree/695dd12882449179a75845635adfd4b526e89b64). Model credit and licensing remain with AxiomicLabs. [Android app](https://github.com/efebaykaraa/gundem-android) · [Export and runtime source](https://github.com/efebaykaraa/gundem-llms).\n'
    (staging / 'README.md').write_text(card)
    (staging / 'mobile-onnx-catalog.json').write_text(json.dumps(states, indent=2))
    result = api.upload_folder(repo_id=repo, folder_path=staging, commit_message='Publish native-tested original and fine-tuned INT8 ONNX bundles')
    api.create_tag(repo, tag='mobile-v1.1.1', revision=result.oid, exist_ok=True)
    info = api.model_info(repo, revision=result.oid, files_metadata=True)
    for state in states:
        remote = next(s for s in info.siblings if s.rfilename == 'onnx/' + state['file'])
        assert remote.size == state['size'] and remote.lfs.sha256 == state['sha256']
        state['url'] = f'https://huggingface.co/{repo}/resolve/{result.oid}/onnx/{state["file"]}'
    (root / 'published/gpt-x3-150m.json').write_text(json.dumps(states, ensure_ascii=False, indent=2) + '\n')
    print('PUBLISHED GPT-X3 pair:', result.oid, flush=True)


if __name__ == '__main__':
    main()
