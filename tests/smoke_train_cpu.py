#!/usr/bin/env python3
"""Offline two-step LoRA + merge smoke; synthetic data/tiny Llama, NOT a trained LFM."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--uniform-loss', action='store_true')
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from safetensors.torch import load_file
    import pipeline as p
    torch.manual_seed(42)
    label = {'etiketler':['EMEK'],'tur':'SOMUT_GELISME','etki':1,'ciddiyet':2,'aciliyet':1}
    values = [{'id':p.title_id('Smoke headline '+str(i)), 'title':'Smoke headline '+str(i), 'analysis':label,
               'teacher_signature':'synthetic-smoke-test'} for i in range(80)]
    p.write_rows(root / 'labels.jsonl', values)
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator([p.compact(label), (ROOT/'student-system.txt').read_text(), *[r['title'] for r in values]],
                                  trainers.BpeTrainer(vocab_size=256,special_tokens=['[UNK]','[PAD]','[EOS]']))
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
    tok.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    tok.save_pretrained(root/'tiny')
    LlamaForCausalLM(LlamaConfig(vocab_size=len(tok),hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=1024,
        eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)).save_pretrained(root/'tiny')
    subprocess.run([sys.executable,str(ROOT/'pipeline.py'),'prepare','--input',str(root/'labels.jsonl'),'--out',str(root/'dataset')],check=True)
    subprocess.run([sys.executable,str(ROOT/'train.py'),'train','--student',str(root/'tiny'),'--data',str(root/'dataset'),
                    '--out',str(root/'adapter'),'--max-steps','2','--accumulation','1','--cpu','--allow-small',
                    *(['--uniform-loss'] if args.uniform_loss else [])],check=True)
    subprocess.run([sys.executable,str(ROOT/'train.py'),'merge','--adapter',str(root/'adapter'),
                    '--out',str(root/'merged'),'--cpu'],check=True)
    base = load_file(root/'tiny/model.safetensors')
    merged = load_file(root/'merged/model.safetensors')
    adapter = load_file(root/'adapter/adapter_model.safetensors')
    changed = [k for k in base if k in merged and not torch.equal(base[k],merged[k])]
    assert changed and any(torch.count_nonzero(v)>0 for k,v in adapter.items() if 'lora_B' in k)
    print(f'PASS: actual CPU training + merge changed {len(changed)} tensors. This is a synthetic smoke, not LFM training.')

if __name__ == '__main__':
    main()
