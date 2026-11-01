#!/usr/bin/env python3
"""Export our pinned GPT-X3 pair to stateful INT8 ONNX bundles with exact BPE vocab."""
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def vocab_gguf(tokenizer_path, output):
    sys.path.insert(0, str(ROOT.parent.parent / 'vendor/llama.cpp/gguf-py'))
    import gguf
    raw = json.loads(tokenizer_path.read_text())
    vocab = raw['model']['vocab']
    tokens = [f'[UNUSED{i}]' for i in range(65536)]
    types = [gguf.TokenType.UNUSED] * 65536
    for token, index in vocab.items():
        tokens[index] = token
        types[index] = gguf.TokenType.NORMAL
    for token in raw['added_tokens']:
        tokens[token['id']] = token['content']
        types[token['id']] = gguf.TokenType.CONTROL if token['special'] else gguf.TokenType.USER_DEFINED
    assert all(token is not None for token in tokens) and len(tokens) == 65536
    writer = gguf.GGUFWriter(str(output), 'gpt2')
    writer.add_name('GPT-X3 tokenizer only (no model weights)')
    writer.add_context_length(8192)
    writer.add_embedding_length(576)
    writer.add_block_count(24)
    writer.add_head_count(9)
    writer.add_feed_forward_length(1584)
    writer.add_layer_norm_eps(1e-6)
    writer.add_vocab_size(len(tokens))
    writer.add_tokenizer_model('gpt2')
    # DBRX's pre-tokenizer has exactly the published GPT-X3 regex, with normal
    # BPE merges (unlike the Llama3 shortcut that ignores merges).
    writer.add_tokenizer_pre('dbrx')
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges([' '.join(m) if isinstance(m, list) else m for m in raw['model']['merges']])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_pad_token_id(0)
    writer.add_add_bos_token(False)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    import torch
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from peft import PeftModel
    from lab.modeling import load_model, configure_tokenizer
    from onnx_gptx3.step import GPTX3DecodeStep, export
    from onnx_gptx3.runtime import GPTX3OnnxModel
    directory = ROOT / 'workspace/experiments/gpt-x3-150m'
    spec = json.loads((directory / 'model.lock.json').read_text())
    snapshot = Path(snapshot_download(spec['repo'], revision=spec['revision'], local_files_only=True,
        allow_patterns=['*.json', '*.safetensors', '*.py', '*.txt', '*.model', '*.tiktoken', '*.jinja', '*.jinja2']))
    out = ROOT / 'workspace/mobile/gpt-onnx'
    out.mkdir(exist_ok=True)
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True), spec['prompt_style'])
    vocab_gguf(directory / 'adapter/tokenizer.json', out / 'tokenizer.gguf')
    # Test native BPE against the actual HF tokenizer, including Turkish,
    # apostrophes, digits, whitespace, emoji, and all held-out headlines.
    samples = ['İşçiler ücret artışı için greve başladı', '1234567890', "don't I'LL we'Re", '  a\n\n\tb ', '🧑🏽‍💻 Türkiye € 42,5']
    samples += [json.loads(line)['title'] for line in (ROOT / 'workspace/datasets/news-v1/test.jsonl').read_text().splitlines()]
    cli = ROOT.parent.parent / 'build/host-smoke/gundem-tokenizer-smoke'
    for sample in samples:
        result = subprocess.run([str(cli), str(out / 'tokenizer.gguf')], input=sample, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        assert json.loads(result.stdout) == tokenizer.encode(sample, add_special_tokens=False), sample
    print('TOKENIZER PARITY OK:', len(samples), flush=True)
    report = json.loads((ROOT / 'workspace/reports/leaderboard.json').read_text())['results']
    for variant in ('original', 'distilled'):
        target = out / variant
        target.mkdir(exist_ok=True)
        model = load_model(snapshot, torch.float32, revision=spec['revision'])
        model.train()
        if variant == 'distilled':
            model = PeftModel.from_pretrained(model, directory / 'adapter').merge_and_unload(safe_merge=True)
        model.config.use_fla = False
        model.eval()
        config = model.config.to_dict()
        config["_name_or_path"] = spec["repo"]
        (target / 'config.json').write_text(json.dumps(config, indent=2))
        full = target / 'model-fp32.onnx'
        if not full.exists():
            print('EXPORTING ONNX', variant, flush=True)
            export(GPTX3DecodeStep(model).eval(), full)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        runtime = GPTX3OnnxModel(full, model_dir=target, session_options=options)
        ids = tokenizer.encode('<|startoftext|>System: Classify headline.\nUser: İşçiler grevde\nAssistant: ', add_special_tokens=False)
        with torch.inference_mode():
            reference = model(torch.tensor([ids]), use_cache=False).logits[0].numpy()
        states = runtime.new_states()
        worst = 0.
        for position, token in enumerate(ids):
            logits, states = runtime.step([token], position, states)
            worst = max(worst, float(np.abs(logits[0] - reference[position]).max()))
        assert worst < .003, worst
        print('ONNX FP32 PARITY', variant, worst, flush=True)
        del runtime, model
        gc.collect()
        quant = target / 'model.onnx'
        if not quant.exists():
            quantize_dynamic(str(full), str(quant), weight_type=QuantType.QInt8,
                             per_channel=True, op_types_to_quantize=['MatMul'])
        evaluation = next(r for r in report if r['model_id'] == 'gpt-x3-150m' and r['variant'] == variant)
        manifest = {'backend': 'gptx3-onnx-v1', 'version': 1, 'model_id': 'gpt-x3-150m', 'variant': variant,
                    'config': config, 'base': spec, 'evaluation': evaluation, 'model_sha256': sha(quant),
                    'tokenizer_sha256': sha(out / 'tokenizer.gguf'), 'fp32_max_logit_error': worst,
                    'precision': 'INT8 dynamic MatMul; FP32 embeddings and state'}
        (target / 'bundle.json').write_text(json.dumps(manifest, indent=2))
        with zipfile.ZipFile(out / f'gpt-x3-150m-{variant}-ONNX-INT8.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.write(quant, 'model.onnx')
            archive.write(out / 'tokenizer.gguf', 'tokenizer.gguf')
            archive.write(target / 'bundle.json', 'bundle.json')
            license_path = ROOT / 'workspace/releases/models-v1.0.0/models/GPT-X3-150M-Gundem-LoRA/LICENSE'
            archive.write(license_path, 'LICENSE')
        print('BUNDLE READY', variant, flush=True)


if __name__ == '__main__':
    main()
