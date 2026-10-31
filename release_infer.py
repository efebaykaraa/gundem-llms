#!/usr/bin/env python3
"""Load a released Gündem adapter with the same merge and decoding as evaluation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local release folder or HF adapter repository')
    parser.add_argument('--revision', default='main', help='Adapter repository revision')
    parser.add_argument('--headline', action='append', required=True)
    args = parser.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, GenerationConfig
    from peft import PeftModel
    try:
        from gundem_modeling import load_model, configure_tokenizer, template_hash
        from gundem_validation import escape_title, validate_label
    except ModuleNotFoundError:
        from lab.modeling import load_model, configure_tokenizer, template_hash
        from pipeline import escape_title, validate_label
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('The released HF runtime requires a CUDA GPU with BF16 support')
    path = Path(args.model)
    if not path.is_dir():
        path = Path(snapshot_download(args.model, revision=args.revision, allow_patterns=[
            '*.json', '*.safetensors', '*.jinja', '*.txt', '*.model', '*.py']))
    manifest = json.loads((path / 'release-manifest.json').read_text())
    spec = manifest['model']
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(path, trust_remote_code=True),
                                    spec['prompt_style'])
    tokenizer.padding_side = 'left'
    if template_hash(tokenizer) != manifest['evaluation']['chat_template_sha256']:
        raise ValueError('Released tokenizer template does not match the evaluated template')
    model = load_model(spec['repo'], torch.float32, revision=spec['revision'])
    model.train()
    model = PeftModel.from_pretrained(model, path).merge_and_unload(safe_merge=True)
    model.train()
    model = model.to(dtype=torch.bfloat16, device='cuda').eval()
    model.config.use_cache = True
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    eos = sorted(set(eos + [tokenizer.eos_token_id]))
    generation = GenerationConfig(do_sample=False, max_new_tokens=160, repetition_penalty=1.,
                                  use_cache=True, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id)
    prompt = (path / 'student-system.txt').read_text().strip()
    results = []
    for offset in range(0, len(args.headline), 4):
        headlines = args.headline[offset:offset + 4]
        texts = [tokenizer.apply_chat_template([
            {'role': 'system', 'content': prompt}, {'role': 'user', 'content': escape_title(title)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for title in headlines]
        inputs = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            generated = model.generate(**inputs, generation_config=generation)
        for title, tokens in zip(headlines, generated[:, inputs.input_ids.shape[1]:].tolist()):
            stop = next((i for i, token in enumerate(tokens) if token in eos), None)
            content = tokens if stop is None else tokens[:stop]
            row = {'headline': title}
            try:
                if any(token >= len(tokenizer) for token in content):
                    raise ValueError('Generated IDs outside the text tokenizer vocabulary')
                raw = tokenizer.decode(content, skip_special_tokens=True).strip()
                row['raw_text'] = raw
                if stop is None:
                    raise ValueError('Truncated or unfinished output')
                row['prediction'] = validate_label(json.loads(raw))
            except (ValueError, TypeError, KeyError) as error:
                row['error'] = str(error)
            results.append(row)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
