#!/usr/bin/env python3
"""Merge first; convert HF weights to GGUF, quantize Q4_K_M, and record SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--llama-dir', required=True, type=Path)
    p.add_argument('--quantizer', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    converter = args.llama_dir / 'convert_hf_to_gguf.py'
    if not converter.is_file() or not args.quantizer.is_file() or not (args.model / 'config.json').is_file():
        p.error('Need merged HF model, llama.cpp converter, and built llama-quantize executable')
    if args.out.exists():
        p.error('Output exists; choose a new filename')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gundem-gguf-', dir=args.out.parent) as tmp:
        full = Path(tmp) / 'model-f16.gguf'
        quant = Path(tmp) / 'model-q4.gguf'
        subprocess.run([sys.executable, str(converter.resolve()), str(args.model.resolve()), '--outfile', str(full), '--outtype', 'f16'], check=True)
        subprocess.run([str(args.quantizer.resolve()), str(full), str(quant), 'Q4_K_M'], check=True)
        with quant.open('rb') as f:
            if f.read(4) != b'GGUF':
                raise ValueError('Quantizer output is not GGUF')
        shutil.move(str(quant), args.out)
    sha = hashlib.sha256()
    with args.out.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(chunk)
    metadata = {'name': args.out.name, 'sha256': sha.hexdigest(), 'size': args.out.stat().st_size,
                'quantization': 'Q4_K_M', 'merged_model': str(args.model.resolve())}
    args.out.with_suffix(args.out.suffix + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()
