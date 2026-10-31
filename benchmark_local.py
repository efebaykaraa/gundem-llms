#!/usr/bin/env python3
"""Compare original and merged HF models sequentially, without API servers."""
import argparse
import gc
import json
from pathlib import Path

import pipeline


class LocalClient:
    def __init__(self, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for this local benchmark')
        self.torch = torch
        self.config = config
        self.completed = 0
        self.tokenizer = AutoTokenizer.from_pretrained(config['model'], trust_remote_code=True, local_files_only=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model'], trust_remote_code=True, local_files_only=True, dtype=dtype).to('cuda').eval()
        # Override model-specific sampling defaults so both models use the same decoding.
        self.generation = GenerationConfig(
            do_sample=False, max_new_tokens=config['max_tokens'], repetition_penalty=1.0,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            use_cache=True)

    def generate(self, title, prompt):
        messages = [{'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': pipeline.escape_title(title)}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').to('cuda')
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, generation_config=self.generation)
        tokens = result[0, inputs['input_ids'].shape[1]:]
        eos = self.generation.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        if not len(tokens) or tokens[-1].item() not in eos:
            raise ValueError('Truncated or unfinished output')
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

    def warmup(self, title, prompt):
        # Invalid JSON during warmup must not prevent measuring invalid outputs.
        try:
            self.generate(title, prompt)
        except ValueError:
            pass

    def classify(self, title, prompt):
        try:
            return pipeline.validate_label(json.loads(self.generate(title, prompt))), {}
        finally:
            self.completed += 1
            print(f'  {self.completed}/{self.config["count"]}', flush=True)

    def close(self):
        del self.model
        del self.tokenizer
        gc.collect()
        self.torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data-deepseek/prepared', type=Path)
    parser.add_argument('--new-manifest', default='data-deepseek/merged/distill-manifest.json', type=Path)
    parser.add_argument('--old-model', type=Path)
    parser.add_argument('--new-model', default='data-deepseek/merged', type=Path)
    parser.add_argument('--out', default='data-deepseek/benchmark-local.json', type=Path)
    parser.add_argument('--max-tokens', default=160, type=int)
    parser.add_argument('--warmup', default=1, type=int)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.warmup < 0:
        parser.error('max-tokens must be positive; warmup must be nonnegative')
    trained = json.loads(args.new_manifest.read_text())
    if args.old_model is None:
        from huggingface_hub import snapshot_download
        args.old_model = Path(snapshot_download(
            trained['student'], revision=trained['revision'], local_files_only=True,
            allow_patterns=['*.json', '*.safetensors', '*.py', '*.txt',
                            '*.model', '*.tiktoken', '*.jinja', '*.jinja2']))
    for path in (args.old_model, args.new_model):
        if not (path / 'config.json').is_file():
            parser.error(f'Model directory missing: {path}')
    count = len(pipeline.rows(args.data / 'test.jsonl'))
    configs = [dict(model=str(path.resolve()), temperature=0, max_tokens=args.max_tokens,
                    response_format='none', extra_body={}, count=count)
               for path in (args.old_model, args.new_model)]
    args.backend = 'transformers CUDA; greedy; no constrained JSON decoding; model loading excluded'
    pipeline.benchmark(args, configs=configs, client_factory=LocalClient)


if __name__ == '__main__':
    main()
