"""Uniform batched HF/CUDA evaluation, in one disposable process per variant."""
import argparse
import gc
import json
import time
import importlib.metadata
from pathlib import Path

import pipeline
from .common import read_json, write_json, sha, validate_dataset, validate_training
from .modeling import configure_tokenizer, load_model, template_hash


def runtime_versions():
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'triton', 'flash-linear-attention', 'fla-core'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def fingerprint(spec, dataset, settings, adapter=None):
    value = {'spec': spec, 'test_sha256': dataset['split_sha256']['test'],
             'student_prompt_sha256': dataset['student_prompt_sha256'], 'settings': settings,
             'adapter_manifest': sha(Path(adapter) / 'distill-manifest.json') if adapter else None,
             'adapter_weights': sha(Path(adapter) / 'adapter_model.safetensors') if adapter else None}
    value['implementation'] = {name: sha(Path(__file__).parent / name)
                               for name in ('evaluate.py', 'modeling.py')}
    value['runtime_versions'] = runtime_versions()
    return pipeline.digest(pipeline.compact(value))


def summarize(predictions):
    if not predictions:
        raise ValueError('No predictions')
    matches = types = invalid = tp = fp = fn = 0
    errors = {k: 0 for k in pipeline.FIELDS[2:]}
    for row in predictions:
        truth = row['truth']
        if 'prediction' not in row:
            invalid += 1
            fn += len(truth['etiketler'])
            for k in errors:
                errors[k] += 3
            continue
        pred = row['prediction']
        actual, expected = set(pred['etiketler']), set(truth['etiketler'])
        tp += len(actual & expected)
        fp += len(actual - expected)
        fn += len(expected - actual)
        matches += pred == truth
        types += pred['tur'] == truth['tur']
        for k in errors:
            errors[k] += abs(pred[k] - truth[k])
    n = len(predictions)
    times = sorted(row['seconds'] for row in predictions)
    return {'count': n, 'invalid': invalid, 'exact_match': matches / n,
            'topic_micro_f1': 2 * tp / max(1, 2 * tp + fp + fn), 'type_accuracy': types / n,
            'score_mae': {k: v / n for k, v in errors.items()},
            'latency_seconds': {'mean': sum(times) / n, 'p50': times[(n - 1) // 2],
                                'p95': times[min(n - 1, (95 * n + 99) // 100 - 1)]}}


def evaluate(spec, dataset, settings, adapter, out, training):
    from train import configure_hub_download, download_student
    configure_hub_download()
    import torch
    from transformers import AutoTokenizer, GenerationConfig
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Suite requires a CUDA GPU with bfloat16 support')
    manifest, splits = validate_dataset(dataset)
    if adapter:
        trained = validate_training(adapter, manifest, training, spec)
    else:
        trained = None
    model_path = download_student(spec['repo'], spec['revision'], 'plain')
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(model_path, trust_remote_code=True),
                                    spec['prompt_style'])
    tokenizer.padding_side = 'left'
    if trained and trained.get('chat_template_sha256') and trained['chat_template_sha256'] != template_hash(tokenizer):
        raise ValueError('Trained and evaluated chat templates differ')
    # Merge in float32 CPU memory, then evaluate all models in the same bfloat16.
    # No extra full-weight copy is written to disk.
    model = load_model(model_path, torch.float32 if adapter else torch.bfloat16,
                       revision=spec['revision'])
    model.train()  # Clear custom-model inference caches before attaching/merging LoRA.
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload(safe_merge=True)
        model.train()
    model = model.to(dtype=torch.bfloat16, device='cuda').eval()
    model.config.use_cache = True
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    eos = sorted(set(eos + [tokenizer.eos_token_id]))
    generation = GenerationConfig(do_sample=False, max_new_tokens=settings['max_new_tokens'],
                                  repetition_penalty=1.0, use_cache=True,
                                  eos_token_id=eos, pad_token_id=tokenizer.pad_token_id)
    records = splits['test']

    def generate(batch):
        texts = [tokenizer.apply_chat_template(
            [{'role': 'system', 'content': r['messages'][0]['content']},
             {'role': 'user', 'content': pipeline.escape_title(r['title'])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in batch]
        inputs = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=generation)
        return output[:, inputs.input_ids.shape[1]:].tolist()

    print(f"Evaluation: {spec['id']} / {'distilled' if adapter else 'original'}", flush=True)
    for _ in range(settings['warmup']):
        generate(records[:settings['batch_size']])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    predictions = []
    batch_seconds = []
    for offset in range(0, len(records), settings['batch_size']):
        batch = records[offset:offset + settings['batch_size']]
        torch.cuda.synchronize()
        started = time.perf_counter()
        results = generate(batch)  # Runtime/OOM exceptions abort; never fabricate 0% scores.
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        batch_seconds.append(elapsed)
        for row, tokens in zip(batch, results):
            result = {'id': row['id'], 'title': row['title'],
                      'truth': pipeline.validate_label(json.loads(row['messages'][-1]['content'])),
                      'seconds': elapsed / len(batch), 'batch_seconds': elapsed,
                      'batch_size': len(batch)}
            stop = next((i for i, token in enumerate(tokens) if token in eos), None)
            content = tokens if stop is None else tokens[:stop]
            result['output_token_count'] = len(content)
            try:
                if any(token >= len(tokenizer) for token in content):
                    raise ValueError('Model generated IDs outside the text tokenizer vocabulary')
                text = tokenizer.decode(content, skip_special_tokens=True).strip()
                result['raw_text'] = text
                if stop is None:
                    raise ValueError('Truncated or unfinished output')
                result['prediction'] = pipeline.validate_label(json.loads(text))
            except (ValueError, TypeError, KeyError) as error:
                result['error'] = str(error)
            predictions.append(result)
        # Crash-safe partial predictions are useful for diagnosis, but never reused as full results.
        pipeline.write_rows(str(out) + '.partial.predictions.jsonl', predictions)
        print(f'  {len(predictions)}/{len(records)}', flush=True)
    report = summarize(predictions)
    report.update(model=spec['repo'], revision=spec['revision'], model_id=spec['id'],
                  label=spec['label'], variant='distilled' if adapter else 'original',
                  fingerprint=fingerprint(spec, manifest, settings, adapter),
                  test_sha256=manifest['split_sha256']['test'], settings=settings,
                  prompt_style=spec['prompt_style'], chat_template_sha256=template_hash(tokenizer),
                  backend='transformers CUDA bfloat16; greedy; no JSON grammar',
                  timing='Amortized batch inference seconds per headline; not serial request latency',
                  total_inference_seconds=sum(batch_seconds),
                  headlines_per_second=len(records) / sum(batch_seconds),
                  parameters=model.num_parameters(), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  versions=runtime_versions())
    pipeline.write_rows(str(out) + '.predictions.jsonl', predictions)
    write_json(out, report)
    Path(str(out) + '.partial.predictions.jsonl').unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--spec', required=True, type=Path)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--adapter', type=Path)
    a = p.parse_args()
    plan = read_json(a.plan)
    evaluate(read_json(a.spec), a.dataset, plan['evaluation'], a.adapter, a.out, plan['training'])


if __name__ == '__main__':
    main()
