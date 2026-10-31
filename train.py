#!/usr/bin/env python3
"""Response distillation: train LFM on Qwen's JSON answers with assistant-only LoRA loss."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from pipeline import compact, digest, escape_title, rows, validate_label
from controls import SavedExit
from progress import StageProgress, download_tqdm, progress_mode


def configure_hub_download():
    # Must run before Transformers/Hub imports: Hub reads these values at import.
    # The Xet reconstruction path stalls on this host; regular HTTPS also uses
    # the Hub cache and supports resuming interrupted HTTP downloads.
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '30')


def download_student(student, revision, mode):
    if Path(student).is_dir():
        return student
    configure_hub_download()
    from huggingface_hub import snapshot_download
    print(f'Model indiriliyor / önbellek kontrol ediliyor: {student}', flush=True)
    path = snapshot_download(repo_id=student, revision=revision,
                             allow_patterns=['*.json', '*.safetensors', '*.py', '*.txt',
                                             '*.model', '*.tiktoken', '*.jinja', '*.jinja2'],
                             max_workers=2,
                             tqdm_class=download_tqdm(mode))
    print('Model dosyaları hazır.', flush=True)
    return path


def stop_requested():
    path = os.environ.get('DISTILL_STOP_FILE')
    return bool(path and Path(path).exists())


def save_and_stop(args, state, control, **kwargs):
    if stop_requested():
        control.should_save = True
        control.should_training_stop = True
        control.should_evaluate = True
    return control


def encode_example(tokenizer, row, max_length):
    messages = row['messages']
    if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError('Expected exactly system/user/assistant messages')
    if messages[1]['content'] != escape_title(row['title']):
        raise ValueError('Student user input must be exactly the escaped headline')
    target = compact(validate_label(json.loads(messages[-1]['content'])))
    # Render exactly the inference prefix, including LFM's non-thinking suffix.
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if not isinstance(prefix, list) or any(type(token) is not int for token in prefix):
        raise ValueError('Chat template must return a flat token-id list')
    answer = tokenizer(target, add_special_tokens=False)['input_ids']
    eos = tokenizer.eos_token_id
    if type(eos) is not int:
        raise ValueError('Tokenizer must expose a single EOS token id')
    if not answer or answer[-1] != eos:
        answer = answer + [eos]
    if len(prefix) + len(answer) > max_length:
        return None  # Never train on truncated JSON or silently truncate the headline.
    return {'input_ids': list(prefix) + list(answer), 'attention_mask': [1] * (len(prefix) + len(answer)),
            'labels': [-100] * len(prefix) + list(answer)}


def check_splits(data):
    seen_ids, seen_groups = set(), set()
    for split in ('train', 'validation', 'test'):
        values = data[split]
        ids = {x['id'] for x in values}
        groups = {x['split_group'] for x in values}
        if len(ids) != len(values) or seen_ids & ids or seen_groups & groups:
            raise ValueError('Duplicate or leaked headline/event group across dataset splits')
        seen_ids |= ids
        seen_groups |= groups


def train(args):
    # Do not import CUDA/Transformers just to use collect/label/prepare or inspect data.
    configure_hub_download()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    from huggingface_hub import HfApi
    from lab.modeling import configure_tokenizer, load_model, template_hash, adapt_direct_weight_lora

    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required by default; --cpu is for small smoke tests')
    directory = Path(args.data)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    data = {split: rows(directory / (split + '.jsonl')) for split in ('train', 'validation', 'test')}
    check_splits(data)
    for split in data:
        if hashlib.sha256((directory / (split + '.jsonl')).read_bytes()).hexdigest() != manifest['split_sha256'][split]:
            raise ValueError('Prepared dataset changed; run prepare again')
    if not data['train'] or not data['validation']:
        raise ValueError('Training and validation splits must both contain examples')
    if not data['test'] and not args.allow_small:
        raise ValueError('A held-out test split is required for the final old/new benchmark; collect more labels')
    if len(data['train']) < 32 and not args.allow_small:
        raise ValueError('Need >=32 training examples; --allow-small enables a pipeline smoke test')
    for values in data.values():
        for row in values:
            prompt = row['messages'][0]['content']
            if digest(prompt) != manifest['student_prompt_sha256'] or len(prompt) > 512:
                raise ValueError('Student prompt mismatch or unexpectedly long prompt (>512 chars)')
    set_seed(args.seed)
    student = args.student
    local = Path(student).is_dir()
    if local:
        student = str(Path(student).resolve())
    out = Path(args.out)
    plan_path = out / 'training-plan.json'
    previous_plan = json.loads(plan_path.read_text()) if args.resume and plan_path.is_file() else None
    if previous_plan and previous_plan['student'] == student and args.revision == 'main':
        revision = previous_plan['revision']
    else:
        revision = None if local else HfApi().model_info(student, revision=args.revision).sha
    plan = {'student': student, 'revision': revision, 'dataset_manifest': manifest,
            'hyperparameters': {k: getattr(args, k) for k in ('epochs','learning_rate','rank','accumulation','max_length','seed','max_steps')}}
    prompt_style = getattr(args, 'prompt_style', 'native')
    if prompt_style != 'native':
        plan['prompt_style'] = prompt_style
    if getattr(args, 'uniform_loss', False):
        from lab.loss import LOSS_CONTRACT
        plan['loss_contract'] = LOSS_CONTRACT
    if previous_plan and previous_plan != plan:
        raise ValueError('Checkpoint dataset/model/training settings changed; choose a new output directory')
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError('Output directory is not empty; use a new directory or --resume CHECKPOINT')
    out.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    common = {'trust_remote_code': True, **({'revision': revision} if revision else {})}
    mode = progress_mode(getattr(args, 'progress', 'auto'))
    model_path = download_student(student, revision, mode)
    with StageProgress('Tokenizer yükleniyor', mode):
        tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(model_path, **common), prompt_style)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer has no pad/EOS token')
    encoded = {}
    for split in ('train', 'validation'):
        examples = [encode_example(tokenizer, row, args.max_length) for row in data[split]]
        encoded[split] = [row for row in examples if row is not None]
        print(json.dumps({'split': split, 'kept': len(encoded[split]), 'dropped_overlength': len(examples) - len(encoded[split])}), flush=True)
        if not encoded[split]:
            raise ValueError(f'All {split} examples exceeded max-length')
    bf16 = not args.cpu and torch.cuda.is_bf16_supported()
    dtype = torch.float32 if args.cpu else torch.bfloat16 if bf16 else torch.float16
    with StageProgress('Model RAM’e yükleniyor', mode):
        model = load_model(model_path, dtype=dtype, **common)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', target_modules='all-linear',
                           r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias='none'))
    model = adapt_direct_weight_lora(model)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    def collate(batch):
        size = max(len(row['input_ids']) for row in batch)
        return {key: torch.tensor([row[key] + [pad] * (size - len(row[key])) for row in batch], dtype=torch.long)
                for key, pad in [('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}

    options = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accumulation, learning_rate=args.learning_rate,
        warmup_steps=0.05, weight_decay=0.01, logging_steps=1, save_strategy='epoch', eval_strategy='epoch',
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        save_total_limit=2, bf16=bf16, fp16=not args.cpu and not bf16, use_cpu=args.cpu,
        gradient_checkpointing=bool(getattr(model, 'supports_gradient_checkpointing', False)),
        gradient_checkpointing_kwargs={'use_reentrant': False},
        report_to=[], remove_unused_columns=False, seed=args.seed, dataloader_num_workers=0,
        disable_tqdm=mode in ('off', 'plain'),
    )
    class SaveAndStopCallback(TrainerCallback):
        on_step_end = staticmethod(save_and_stop)

    if stop_requested():
        raise SavedExit()
    with StageProgress('Eğitim hazırlanıyor / GPU’ya aktarılıyor' if not args.cpu else 'CPU eğitimi hazırlanıyor', mode):
        from lab.loss import assistant_causal_loss, LOSS_CONTRACT
        trainer = Trainer(model=model, args=options, train_dataset=encoded['train'], eval_dataset=encoded['validation'],
                          data_collator=collate, processing_class=tokenizer, callbacks=[SaveAndStopCallback()],
                          compute_loss_func=assistant_causal_loss if getattr(args, 'uniform_loss', False) else None)
        if getattr(args, 'uniform_loss', False):
            trainer.model_accepts_loss_kwargs = False
            trainer.loss_is_scaled_for_ga = True
    print('Eğitim başlıyor; model GPU’ya aktarılmış durumda.' if not args.cpu else 'CPU eğitimi başlıyor.', flush=True)
    trainer.train(resume_from_checkpoint=args.resume or None)
    if stop_requested():
        # Trainer saved model, optimizer, scheduler and RNG state at the step boundary.
        # Do not mark this interrupted run as a completed model.
        raise SavedExit()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(out)
    report = trainer.evaluate()
    run = {'student': student, 'revision': revision, 'prompt_sha256': manifest['student_prompt_sha256'],
           'dataset_manifest': manifest, 'metrics': report,
           'hyperparameters': {k: getattr(args, k) for k in ('epochs','learning_rate','rank','accumulation','max_length','seed','max_steps')}, 'method': 'response-distillation + assistant-only LoRA SFT',
           'versions': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                        'peft': __import__('peft').__version__},
           'split_hashes': {split: hashlib.sha256((directory / (split + '.jsonl')).read_bytes()).hexdigest() for split in data}}
    run['prompt_style'] = prompt_style
    run['chat_template_sha256'] = template_hash(tokenizer)
    run['gradient_checkpointing'] = options.gradient_checkpointing
    run['loss_contract'] = LOSS_CONTRACT if getattr(args, 'uniform_loss', False) else 'provider-native'
    (out / 'distill-manifest.json').write_text(json.dumps(run, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def merge(args):
    configure_hub_download()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftConfig, PeftModel
    from lab.modeling import load_model
    adapter = Path(args.adapter)
    manifest = json.loads((adapter / 'distill-manifest.json').read_text(encoding='utf-8'))
    config = PeftConfig.from_pretrained(adapter)
    common = {'trust_remote_code': True}
    if manifest.get('revision'):
        common['revision'] = manifest['revision']
    base = load_model(config.base_model_name_or_path, dtype=torch.float32 if args.cpu else torch.float16, **common)
    model = PeftModel.from_pretrained(base, adapter)
    model = model.merge_and_unload()
    model.config.use_cache = True
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Merged output directory must be empty')
    model.save_pretrained(out, safe_serialization=True, max_shard_size='2GB')
    AutoTokenizer.from_pretrained(adapter, trust_remote_code=True).save_pretrained(out)
    (out / 'distill-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Merged HF model saved: {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train'); t.add_argument('--data', required=True); t.add_argument('--out', required=True)
    t.add_argument('--student', default='LiquidAI/LFM3-1.7B'); t.add_argument('--revision', default='main')
    t.add_argument('--epochs', type=float, default=3); t.add_argument('--learning-rate', type=float, default=2e-4)
    t.add_argument('--rank', type=int, default=16); t.add_argument('--accumulation', type=int, default=8)
    t.add_argument('--max-length', type=int, default=512); t.add_argument('--seed', type=int, default=42)
    t.add_argument('--max-steps', type=int, default=-1); t.add_argument('--resume')
    t.add_argument('--progress', choices=['auto', 'bar', 'plain', 'off'], default='auto')
    t.add_argument('--prompt-style', choices=['native', 'auto', 'plain', 'task-chatml'], default='native')
    t.add_argument('--uniform-loss', action='store_true', help='Use common assistant-token-normalized causal loss across architectures')
    t.add_argument('--cpu', action='store_true'); t.add_argument('--allow-small', action='store_true'); t.set_defaults(run=train)
    m = sub.add_parser('merge'); m.add_argument('--adapter', required=True); m.add_argument('--out', required=True)
    m.add_argument('--cpu', action='store_true'); m.set_defaults(run=merge)
    args = parser.parse_args()
    try:
        args.run(args)
    except SavedExit:
        print('Eğitim durduruldu; mevcut checkpointler korundu.', file=sys.stderr)
        return 130
    except ImportError as error:
        print(f'Training dependencies missing: {error}. Install requirements-train.txt in a separate venv.', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
