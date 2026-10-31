#!/usr/bin/env python3
"""RSS -> teacher JSON labels -> headline-only SFT data. Python 3.10+, stdlib only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import datetime as dt
import email.utils
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from progress import TeacherProgress
from controls import SavedExit, stop_session

HERE = Path(__file__).resolve().parent
_PROGRESS = None
TOPICS = ['TKP', 'EMEK', 'TURKIYE', 'ULUSLARARASI', 'KUBA', 'EKONOMI', 'EYLEM', 'DIGER']
TYPES = ['ACIKLAMA', 'CAGRI', 'SOMUT_GELISME', 'YORUM']
FIELDS = ['etiketler', 'tur', 'etki', 'ciddiyet', 'aciliyet']
SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': FIELDS,
    'properties': {
        'etiketler': {'type': 'array', 'items': {'type': 'string', 'enum': TOPICS}, 'minItems': 1, 'maxItems': 7, 'uniqueItems': True},
        'tur': {'type': 'string', 'enum': TYPES},
        **{key: {'type': 'integer', 'minimum': 0, 'maximum': 3} for key in FIELDS[2:]},
    },
}

def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def normalize(title):
    return ' '.join(unicodedata.normalize('NFC', title).casefold().split())

def title_id(title):
    return digest(normalize(title))

def escape_title(title):
    # Same escaping as Android's escapeControlTokens; no article metadata is added.
    return title.replace('<', '＜').replace('>', '＞').replace('[INST]', '［INST］').replace('[/INST]', '［/INST］')

def validate_label(value):
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise ValueError('Output must contain exactly the five analysis fields')
    tags = value['etiketler']
    if not isinstance(tags, list) or not tags or any(type(x) is not str or x not in TOPICS for x in tags):
        raise ValueError('Invalid topic list')
    if len(set(tags)) != len(tags) or ('DIGER' in tags and len(tags) != 1):
        raise ValueError('Duplicate or mixed DIGER topic')
    if value['tur'] not in TYPES or any(type(value[k]) is not int or not 0 <= value[k] <= 3 for k in FIELDS[2:]):
        raise ValueError('Invalid type or score')
    return {k: (sorted(tags, key=TOPICS.index) if k == 'etiketler' else value[k]) for k in FIELDS}

def rows(path):
    p = Path(path)
    if not p.exists():
        return []
    result = []
    for n, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f'{p}:{n}: broken JSONL; repair the partial line before resuming') from e
    return result

def write_rows(path, values):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(''.join(compact(x) + '\n' for x in values), encoding='utf-8')
    temporary.replace(p)

def append(path, row):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(compact(row) + '\n')
        f.flush()
        os.fsync(f.fileno())

def log(path, event, **detail):
    row = {'time': dt.datetime.now(dt.timezone.utc).isoformat(), 'event': event, **detail}
    if _PROGRESS is None:
        print(compact(row), file=sys.stderr, flush=True)
    if path:
        append(path, row)

def date_value(text):
    if not text:
        return None
    try:
        try:
            date = email.utils.parsedate_to_datetime(text)
        except (ValueError, TypeError):
            date = dt.datetime.fromisoformat(text.strip().replace('Z', '+00:00'))
        if date.tzinfo is None:
            return None  # Do not guess timezone or replace publication time with fetch time.
        return date.astimezone(dt.timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError):
        return None

def parse_feed(data, source, fetched=None):
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise ValueError('DTD/entities are not supported')
    root = ET.fromstring(data)
    fetched = fetched or dt.datetime.now(dt.timezone.utc).isoformat()
    result = []
    for item in root.iter():
        if item.tag.rsplit('}', 1)[-1] not in ('item', 'entry'):
            continue
        children = {child.tag.rsplit('}', 1)[-1]: child for child in item}
        def text(key):
            node = children.get(key)
            return ''.join(node.itertext()).strip() if node is not None else ''
        title = ' '.join(html.unescape(re.sub('<[^>]+>', '', text('title'))).split())
        if not title:
            continue
        link = text('link')
        if not link:
            for node in item:
                if node.tag.rsplit('}', 1)[-1] == 'link' and node.get('rel', 'alternate') == 'alternate':
                    link = node.get('href', '')
                    break
        result.append({'id': title_id(title), 'title': title, 'source': source['id'],
                       'url': urllib.parse.urljoin(source['url'], link),
                       'published': date_value(text('pubDate') or text('published') or text('updated')),
                       'fetched': fetched})
    return result

def get_url(url, limit=12 * 1024 * 1024):
    if urllib.parse.urlsplit(url).scheme != 'https':
        raise ValueError('RSS URL must use HTTPS')
    req = urllib.request.Request(url, headers={'User-Agent': 'Gundem-Distill/0.9 (RSS headlines)', 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(req, timeout=30) as response:
        if urllib.parse.urlsplit(response.url).scheme != 'https':
            raise ValueError('RSS redirect downgraded HTTPS')
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('RSS exceeds 12 MiB')
    return data

def collect(args):
    by_id = {r['id']: r for r in rows(args.out)}
    initial = len(by_id)
    sources = json.loads(Path(args.feeds).read_text(encoding='utf-8'))
    failures = 0
    for source in sources:
        if args.count and len(by_id) >= args.count:
            break
        start = time.monotonic()
        try:
            found = parse_feed(get_url(source['url']), source)
            if args.per_feed:
                found = found[:args.per_feed]
            before = len(by_id)
            for row in found:
                # Keep first provenance for identical normalized headlines across publishers.
                by_id.setdefault(row['id'], row)
                if args.count and len(by_id) >= args.count:
                    break
            log(args.log, 'rss_completed', source=source['id'], count=len(found), added=len(by_id) - before,
                seconds=round(time.monotonic() - start, 3))
        except Exception as e:
            failures += 1
            log(args.log, 'rss_error', source=source['id'], error=type(e).__name__)
    data = sorted(by_id.values(), key=lambda r: (r.get('published') or '', r['id']), reverse=True)
    write_rows(args.out, data)
    if args.count and len(data) < args.count:
        log(args.log, 'collection_target_not_reached', requested=args.count, unique=len(data),
            missing=args.count - len(data),
            message='RSS sources exhausted; add more feeds or collect again when new headlines are published.')
    if not data:
        raise ValueError('No usable headlines collected')
    log(args.log, 'collection_ready', unique=len(data), added=len(data) - initial,
        requested=args.count, failed_sources=failures, out=str(args.out))

class NoApiRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward an Authorization header to a redirected host.


def teacher_options(parser):
    parser.add_argument('--provider', choices=['qwen', 'deepseek'], default='qwen')
    parser.add_argument('--config', help='Teacher config (default: teacher.json or teacher-deepseek.json)')
    parser.add_argument('--teacher-model', help='Override teacher model ID')
    parser.add_argument('--api-key-file', help='Plain token file; overrides configured file/environment')


def load_teacher_config(args):
    provider = getattr(args, 'provider', 'qwen')
    path = getattr(args, 'config', None) or HERE / ('teacher-deepseek.json' if provider == 'deepseek' else 'teacher.json')
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if getattr(args, 'teacher_model', None):
        config['model'] = args.teacher_model
    if getattr(args, 'api_key_file', None):
        config['api_key_file'] = args.api_key_file
    return config


def api_token(config):
    if config.get('api_key_file'):
        path = Path(config['api_key_file']).expanduser()
        try:
            token = path.read_text(encoding='utf-8').strip()
        except (OSError, UnicodeError):
            raise ValueError(f'API anahtar dosyası okunamadı: {path}. Dosyaya yalnız token yazın.') from None
        if not token or any(char.isspace() for char in token):
            raise ValueError('API anahtar dosyası boş veya geçersiz; dosyaya yalnız tek satır token yazın.')
        return token
    return os.environ.get(config.get('api_key_env', 'QWEN_API_KEY'), '')


class ChatClient:
    def __init__(self, config):
        self.config = config
        self.opener = urllib.request.build_opener(NoApiRedirect())
        self.url = config['base_url'].rstrip('/') + '/chat/completions'
        u = urllib.parse.urlsplit(self.url)
        if u.scheme not in ('https', 'http') or not u.hostname or u.username or u.password:
            raise ValueError('Invalid API base URL')
        self.token = api_token(config)

    def auth_headers(self):
        return {'Authorization': 'Bearer ' + self.token} if self.token else {}

    def classify(self, title, prompt):
        c = self.config
        payload = dict(c.get('extra_body', {}))
        payload.update(model=c['model'], messages=[{'role': 'system', 'content': prompt},
                       {'role': 'user', 'content': escape_title(title)}],
                       temperature=c.get('temperature', 0), max_tokens=c.get('max_tokens', 512), stream=False)
        fmt = c.get('response_format', 'json_schema')
        if fmt == 'json_schema':
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'news_analysis', 'strict': True, 'schema': SCHEMA}}
        elif fmt == 'json_object':
            payload['response_format'] = {'type': 'json_object'}
        elif fmt != 'none':
            raise ValueError('response_format must be json_schema, json_object or none')
        headers = {'Content-Type': 'application/json', **self.auth_headers()}
        req = urllib.request.Request(self.url, data=compact(payload).encode('utf-8'), headers=headers)
        with self.opener.open(req, timeout=c.get('timeout', 180)) as response:
            body = response.read(2 * 1024 * 1024 + 1)
        if len(body) > 2 * 1024 * 1024:
            raise ValueError('API response too large')
        value = json.loads(body)
        choice = value['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Truncated or unfinished teacher output')
        text = choice['message'].get('content')
        if not isinstance(text, str):
            raise ValueError('Missing final content; reasoning text is not a label')
        return validate_label(json.loads(text)), value.get('usage', {})

OUTPUT_TEMPLATE = {'etiketler': [], 'tur': None, 'etki': None, 'ciddiyet': None, 'aciliyet': None}


def teacher_system(prompt):
    return (prompt.rstrip() + "\n\nAşağıdaki JSON şablonunun yalnız boş değerlerini doldur. "
            "Anahtarları koru; alan ekleme veya çıkarma. [] yerine ilgili etiketleri, "
            "null yerine geçerli türü veya 0–3 puanı yaz. Çıktıda boş/null alan kalmasın. "
            "Yalnız doldurulmuş JSON döndür.\n" + json.dumps(OUTPUT_TEMPLATE, ensure_ascii=False, indent=2))


def teacher_signature(config, criteria_prompt):
    return digest(compact({'model': config['model'], 'prompt': criteria_prompt, 'schema': SCHEMA,
                          'sampling': {k: config.get(k) for k in ('temperature','max_tokens','extra_body','response_format')}}))


def label(args):
    with stop_session() as control:
        return _label(args, control)


def _label(args, control):
    global _PROGRESS
    config = getattr(args, '_teacher_config', None) or load_teacher_config(args)
    raw_prompt = Path(args.prompt).read_text(encoding='utf-8')
    prompt = teacher_system(raw_prompt)
    signature = teacher_signature(config, raw_prompt)
    previous = rows(args.out)
    if any(r.get('teacher_signature') != signature for r in previous):
        raise ValueError('Teacher/prompt changed; choose a new output file')
    for row in previous:
        validate_label(row['analysis'])
        if row['id'] != title_id(row['title']):
            raise ValueError('Existing label/title identity mismatch')
    done = {r['id'] for r in previous}
    unique = {}
    for article in rows(args.input):
        if article['id'] != title_id(article['title']):
            raise ValueError('Article id/title mismatch')
        unique.setdefault(article['id'], article)
    articles = sorted(unique.values(), key=lambda r: (r.get('published') or '', r['id']), reverse=True)
    if not articles:
        raise ValueError('No usable headlines; collect headlines first')
    reused = len(done & set(unique))
    log_path = args.log or str(args.out) + '.log.jsonl'
    mode = getattr(args, 'progress', 'auto')
    progress = TeacherProgress(len(articles), reused, mode=mode)
    workers = getattr(args, 'workers', 16)
    if workers < 1:
        raise ValueError('workers must be positive')
    pending = [article for article in articles if article['id'] not in done]
    if args.limit:
        pending = pending[:args.limit]
    completed = failed = 0
    output_lock = threading.Lock()

    def process(article):
        nonlocal completed, failed
        if control.requested.is_set():
            return
        client = ChatClient(config)
        start = time.monotonic()
        with output_lock:
            progress.begin(article['id'])
            log(log_path, 'teacher_started', id=article['id'], model=config['model'])
        for attempt in range(args.retries + 1):
            try:
                analysis, usage = client.classify(article['title'], prompt)
            except Exception as e:
                retryable = not isinstance(e, urllib.error.HTTPError) or e.code == 429 or e.code >= 500
                with output_lock:
                    if attempt >= args.retries or not retryable:
                        failed += 1
                        progress.failure()
                        log(log_path, 'teacher_error', id=article['id'], error=type(e).__name__, http=getattr(e, 'code', None))
                        return
                    log(log_path, 'teacher_retry', id=article['id'], attempt=attempt+1, error=type(e).__name__, http=getattr(e, 'code', None))
                    progress.retry(attempt + 1)
                time.sleep(min(2 ** attempt, 8))
                if control.requested.is_set():
                    return
                continue
            with output_lock:
                append(args.out, {**article, 'analysis': analysis, 'teacher_model': config['model'],
                                  'teacher_signature': signature, 'teacher_system_sha256': digest(prompt),
                                  'teacher_format': 'filled-json-v1', 'teacher_usage': usage, 'reviewed': False})
                completed += 1
                progress.success()
                log(log_path, 'teacher_completed', id=article['id'], seconds=round(time.monotonic() - start, 3), usage=usage)
            return

    _PROGRESS = progress if progress.mode != 'off' else None
    try:
        with progress:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for _ in executor.map(process, pending):
                    pass
            log(log_path, 'labeling_ready', completed=completed, failed=failed, reused=reused, workers=workers)
    finally:
        _PROGRESS = None
    if mode != 'off':
        print(f'Etiketler: {args.out} | Ayrıntılı log: {log_path}', file=sys.stderr)
    control.check()
    if failed:
        raise ValueError(f'{failed} labels failed; valid labels saved, rerun to resume')

def prepare(args):
    prompt = Path(args.student_prompt).read_text(encoding='utf-8')
    unique = {}
    for row in rows(args.input):
        if args.reviewed_only and row.get('reviewed') is not True:
            continue
        if row['id'] != title_id(row['title']):
            raise ValueError('Label/title identity mismatch')
        value = validate_label(row['analysis'])
        old = unique.get(row['id'])
        if old and old['analysis'] != value:
            raise ValueError('Conflicting duplicate labels; select one reviewed version')
        unique[row['id']] = {**row, 'analysis': value}
    if not unique:
        raise ValueError('No valid labels for training')
    if len({r.get('teacher_signature') for r in unique.values()}) != 1:
        raise ValueError('Mixed teachers/prompts; keep a single labeling run')
    result = {key: [] for key in ('train', 'validation', 'test')}
    for row in unique.values():
        group = row.get('event_group') or normalize(row['title'])
        split_value = int(digest(str(args.seed) + ':' + group)[:16], 16) % 10000
        split = 'train' if split_value < 8000 else 'validation' if split_value < 9000 else 'test'
        result[split].append({'id': row['id'], 'split_group': group, 'title': row['title'],
                              'messages': [{'role': 'system', 'content': prompt},
                                           {'role': 'user', 'content': escape_title(row['title'])},
                                           {'role': 'assistant', 'content': compact(row['analysis'])}]})
    out = Path(args.out)
    for split, values in result.items():
        write_rows(out / (split + '.jsonl'), sorted(values, key=lambda r: r['id']))
    manifest = {'counts': {k: len(v) for k, v in result.items()}, 'seed': args.seed,
                'student_prompt_sha256': digest(prompt), 'teacher_signature': next(iter(unique.values())).get('teacher_signature'),
                'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                'split_sha256': {split: hashlib.sha256((out / (split + '.jsonl')).read_bytes()).hexdigest() for split in result}}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if any(not v for v in result.values()):
        print('Small dataset: one split is empty. Collect more labels before training/evaluation.', file=sys.stderr)

def evaluate(args):
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    report, predictions = evaluate_records(config, rows(args.input))
    write_rows(str(args.out) + '.predictions.jsonl', predictions)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def evaluate_records(config, data, client=None):
    client = client if client is not None else ChatClient(config)
    matches = types = invalid = tp = fp = fn = 0
    errors = {key: 0 for key in FIELDS[2:]}
    if not data:
        raise ValueError('Empty held-out test set')
    predictions = []
    durations = []
    for row in data:
        messages = row['messages']
        truth = validate_label(json.loads(messages[-1]['content']))
        start = time.monotonic()
        try:
            pred, _ = client.classify(row['title'], messages[0]['content'])
            actual, expected = set(pred['etiketler']), set(truth['etiketler'])
            tp += len(actual & expected); fp += len(actual - expected); fn += len(expected - actual)
            types += pred['tur'] == truth['tur']
            matches += pred == truth
            for key in errors:
                errors[key] += abs(pred[key] - truth[key])
            predictions.append({'id': row['id'], 'truth': truth, 'prediction': pred})
        except Exception as e:
            invalid += 1
            fn += len(truth['etiketler'])
            for key in errors:
                errors[key] += 3  # Invalid predictions cannot improve MAE by being excluded.
            predictions.append({'id': row['id'], 'error': type(e).__name__})
        elapsed = time.monotonic() - start
        durations.append(elapsed)
        predictions[-1]['seconds'] = elapsed
    n = len(data)
    report = {'model': config['model'], 'count': n, 'invalid': invalid, 'exact_match': matches / n,
              'topic_micro_f1': 2 * tp / max(1, 2 * tp + fp + fn), 'type_accuracy': types / n,
              'score_mae': {k: v / n for k, v in errors.items()},
              'latency_seconds': {'mean': sum(durations) / n,
                                  'p50': sorted(durations)[(n - 1) // 2],
                                  'p95': sorted(durations)[min(n - 1, (95 * n + 99) // 100 - 1)]}}
    return report, predictions


def benchmark(args, configs=None, client_factory=None):
    from train import check_splits
    directory = Path(args.data)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    trained = json.loads(Path(args.new_manifest).read_text(encoding='utf-8'))
    if trained['dataset_manifest'] != manifest:
        raise ValueError('Benchmark dataset differs from the new model training manifest')
    data = {split: rows(directory / (split + '.jsonl')) for split in ('train', 'validation', 'test')}
    check_splits(data)
    for split in data:
        if hashlib.sha256((directory / (split + '.jsonl')).read_bytes()).hexdigest() != manifest['split_sha256'][split]:
            raise ValueError('Prepared dataset changed; benchmark cancelled')
    if not data['test']:
        raise ValueError('Empty held-out test set')
    if configs is None:
        configs = [json.loads(Path(path).read_text(encoding='utf-8')) for path in (args.old_config, args.new_config)]
    client_factory = client_factory or ChatClient
    defaults = {'temperature': 0, 'max_tokens': 512, 'response_format': 'json_schema', 'extra_body': {}}
    if any(configs[0].get(k, v) != configs[1].get(k, v) for k, v in defaults.items()):
        raise ValueError('Use identical sampling/output settings for old and new models')
    reports, predictions = {}, {}
    for name, config in zip(('old', 'new'), configs):
        print(f'Benchmark: {name} / {config["model"]}', flush=True)
        client = client_factory(config)
        try:
            if getattr(args, 'warmup', 1):
                for _ in range(args.warmup):
                    row = data['test'][0]
                    getattr(client, 'warmup', client.classify)(row['title'], row['messages'][0]['content'])
            reports[name], predictions[name] = evaluate_records(config, data['test'], client=client)
        finally:
            if hasattr(client, 'close'):
                client.close()
    delta = {key: reports['new'][key] - reports['old'][key]
             for key in ('exact_match', 'topic_micro_f1', 'type_accuracy', 'invalid')}
    delta['score_mae'] = {k: reports['new']['score_mae'][k] - reports['old']['score_mae'][k] for k in FIELDS[2:]}
    delta['mean_seconds'] = reports['new']['latency_seconds']['mean'] - reports['old']['latency_seconds']['mean']
    paired = [{'id': row['id'], 'title': row['title'],
               'truth': json.loads(row['messages'][-1]['content']), 'old': old, 'new': new}
              for row, old, new in zip(data['test'], predictions['old'], predictions['new'])]
    report = {'test_sha256': manifest['split_sha256']['test'], 'count': len(data['test']),
              'warmup_requests_per_model': getattr(args, 'warmup', 1),
              **reports, 'delta_new_minus_old': delta}
    if getattr(args, 'backend', None):
        report['backend'] = args.backend
    if getattr(args, 'comparison', None):
        report['comparison'] = args.comparison
    write_rows(str(args.out) + '.predictions.jsonl', paired)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect'); c.add_argument('--feeds', default=HERE / 'feeds.json'); c.add_argument('--out', required=True)
    c.add_argument('--count', type=int, default=0, help='Target total unique headlines in output, including existing rows (0: all available)')
    c.add_argument('--per-feed', type=int, default=0, help='Maximum entries to inspect per feed (0: all available)')
    c.add_argument('--log'); c.set_defaults(run=collect)
    l = sub.add_parser('label'); l.add_argument('--input', required=True); l.add_argument('--out', required=True)
    teacher_options(l)
    l.add_argument('--prompt', default=HERE / 'teacher-system.txt')
    l.add_argument('--workers', type=int, default=16, help='Concurrent teacher requests (default: 16)')
    l.add_argument('--progress', choices=['auto','bar','plain','off'], default='auto')
    l.add_argument('--retries', type=int, default=2); l.add_argument('--limit', type=int, default=0); l.add_argument('--log'); l.set_defaults(run=label)
    p = sub.add_parser('prepare'); p.add_argument('--input', required=True); p.add_argument('--out', required=True)
    p.add_argument('--student-prompt', default=HERE / 'student-system.txt'); p.add_argument('--reviewed-only', action='store_true')
    p.add_argument('--seed', type=int, default=42); p.set_defaults(run=prepare)
    e = sub.add_parser('evaluate'); e.add_argument('--input', required=True); e.add_argument('--config', required=True)
    e.add_argument('--out', required=True); e.set_defaults(run=evaluate)
    b = sub.add_parser('benchmark', help='Compare old/new models on the held-out test split')
    b.add_argument('--data', required=True); b.add_argument('--new-manifest', required=True)
    b.add_argument('--old-config', required=True); b.add_argument('--new-config', required=True)
    b.add_argument('--warmup', type=int, default=1); b.add_argument('--out', required=True)
    b.set_defaults(run=benchmark)
    args = parser.parse_args()
    if any(getattr(args, name, 0) < 0 for name in ('retries', 'limit', 'per_feed', 'count', 'warmup')):
        parser.error('Counts must be non-negative (count/limit/per-feed=0 means unlimited)')
    if getattr(args, 'workers', 1) < 1:
        parser.error('--workers must be positive')
    try:
        args.run(args)
    except SavedExit:
        print('Kaydedildi ve çıkıldı. Aynı komutla kaldığınız yerden devam edebilirsiniz.', file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        print('Durduruldu; başarılı etiketler kaydedildi.', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
