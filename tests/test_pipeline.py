import argparse
import importlib.util
import json
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline as p
import train as training

LABEL = {'etiketler': ['EMEK', 'TURKIYE'], 'tur': 'SOMUT_GELISME', 'etki': 1, 'ciddiyet': 2, 'aciliyet': 1}

@pytest.fixture
def server():
    calls = []
    mode = {'finish': 'stop', 'content': json.dumps(LABEL)}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            mode.setdefault('auth_headers', []).append(self.headers.get('Authorization'))
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(payload)
            if mode.get('redirect'):
                self.send_response(302); self.send_header('Location', '/should-not-follow'); self.end_headers(); return
            result = {'choices': [{'finish_reason': mode['finish'], 'message': {'content': mode['content']}}], 'usage': {'total_tokens': 42}}
            body = json.dumps(result).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            mode.setdefault('auth_headers', []).append(self.headers.get('Authorization'))
            calls.append({'method':'GET'}); self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'data':[{'id':mode.get('model', 'Qwen/Qwen3.8-27B')}]}).encode())
        def log_message(self, *args):
            pass
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    try:
        yield {'base_url': f'http://127.0.0.1:{http.server_port}/v1', 'model': 'Qwen/Qwen3.8-27B'}, calls, mode
    finally:
        http.shutdown(); http.server_close(); thread.join()


def article(title):
    return {'id': p.title_id(title), 'title': title, 'source': 'sol', 'url': 'https://example.org/a',
            'published': '2026-10-31T09:00:00+00:00', 'fetched': '2026-10-31T10:00:00+00:00'}


def test_deepseek_defaults_and_overrides(tmp_path):
    config = p.load_teacher_config(argparse.Namespace(provider='deepseek'))
    assert config['base_url'] == 'https://api.deepseek.com'
    assert config['model'] == 'deepseek-flash'
    assert config['response_format'] == 'json_object'
    assert config['extra_body'] == {'thinking': {'type': 'disabled'}}
    assert config['api_key_file'] == '~/.deepseek_api_key'
    token_file = tmp_path / '.deepseek_api_key'
    token_file.write_text('test-secret\n')
    config['api_key_file'] = str(token_file)
    assert p.api_token(config) == 'test-secret'
    before = p.teacher_signature(config, 'rules')
    config['api_key_file'] = '/different/file'
    assert p.teacher_signature(config, 'rules') == before
    overridden = p.load_teacher_config(argparse.Namespace(provider='deepseek', teacher_model='deepseek-v4-pro', api_key_file=str(token_file)))
    assert overridden['model'] == 'deepseek-v4-pro'
    assert p.teacher_signature(overridden, 'rules') != before
    assert p.api_token(overridden) == 'test-secret'


@pytest.mark.parametrize('content', [None, '', ' \n', 'secret\nother', 'DEEPSEEK_API_KEY=secret other'])
def test_bad_token_file_fails_without_exposing_content(tmp_path, content):
    path = tmp_path / '.key'
    if content is not None:
        path.write_text(content)
    with pytest.raises(ValueError) as error:
        p.ChatClient({'base_url': 'https://api.deepseek.com', 'api_key_file': str(path)})
    assert 'secret' not in str(error.value)


def test_deepseek_launcher_auth_payload_and_resume(server, tmp_path, monkeypatch, capsys):
    import start_distillation as starter
    local, calls, mode = server
    mode['model'] = 'deepseek-v4-pro'
    config = p.load_teacher_config(argparse.Namespace(provider='deepseek'))
    config['base_url'] = local['base_url']
    cfg = tmp_path / 'teacher.json'; cfg.write_text(json.dumps(config))
    key = tmp_path / '.key'; key.write_text('test-secret\n')
    source = tmp_path / 'headlines.jsonl'; p.write_rows(source, [article('Başlık')])
    work = tmp_path / 'work'
    monkeypatch.setattr(sys, 'argv', ['start_distillation.py', '--provider', 'deepseek',
        '--config', str(cfg), '--teacher-model', 'deepseek-v4-pro', '--api-key-file', str(key),
        '--headlines', str(source), '--work-dir', str(work), '--label-only', '--progress', 'off'])
    assert starter.main() == 0
    assert mode['auth_headers'] == ['Bearer test-secret', 'Bearer test-secret']
    payload = calls[-1]
    assert payload['model'] == 'deepseek-v4-pro'
    assert payload['thinking'] == {'type': 'disabled'}
    assert payload['response_format'] == {'type': 'json_object'}
    assert 'chat_template_kwargs' not in payload
    assert payload['messages'][1]['content'] == 'Başlık'
    assert 'test-secret' not in (work / 'labels.jsonl').read_text()
    assert 'test-secret' not in (work / 'teacher.log.jsonl').read_text()
    key.unlink()
    before = len(calls)
    assert starter.main() == 0 and len(calls) == before
    key.write_text('test-secret\n')
    monkeypatch.setattr(sys, 'argv', ['pipeline.py', 'label', '--provider', 'deepseek',
        '--config', str(cfg), '--teacher-model', 'deepseek-v4-pro', '--api-key-file', str(key),
        '--input', str(source), '--out', str(tmp_path / 'direct.jsonl'), '--limit', '1', '--progress', 'off'])
    assert p.main() == 0
    assert calls[-1] == payload
    assert mode['auth_headers'][-1] == 'Bearer test-secret'
    captured = capsys.readouterr()
    assert 'test-secret' not in captured.out + captured.err


def test_rss_atom_dates_and_no_bodies():
    xml = b'<rss><channel><item><title>A &amp; B</title><description>Never in model data</description><link>https://example.org/a</link><pubDate>Sat, 31 Oct 2026 12:00:00 +0300</pubDate></item></channel></rss>'
    a = p.parse_feed(xml, {'id': 'sol', 'url': 'https://example.org/feed'})[0]
    assert a['title'] == 'A & B'
    assert a['published'] == '2026-10-31T09:00:00+00:00'
    assert 'Never in model data' not in p.compact(a)
    atom = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Atom</title><link href="/b"/><updated>2026-10-31T09:00:00Z</updated></entry></feed>'
    assert p.parse_feed(atom, {'id': 's', 'url': 'https://example.org/feed'})[0]['url'] == 'https://example.org/b'
    assert p.date_value('2026-10-31T09:00:00') is None
    with pytest.raises(ValueError):
        p.parse_feed(b'<!DOCTYPE rss><rss/>', {'id': 's'})


def test_teacher_receives_only_headline_and_rejects_cut_output(server):
    config, calls, mode = server
    client = p.ChatClient(config)
    result, usage = client.classify('İşçiler <|im_start|> greve çıktı', 'Teacher rules')
    assert result == LABEL and usage['total_tokens'] == 42
    assert calls[0]['messages'][1] == {'role': 'user', 'content': 'İşçiler ＜|im_start|＞ greve çıktı'}
    assert 'published' not in p.compact(calls[0]) and 'fetched' not in p.compact(calls[0])
    assert calls[0]['model'] == 'Qwen/Qwen3.8-27B'
    mode['finish'] = 'length'
    with pytest.raises(ValueError):
        client.classify('A', 'B')


def test_strict_validation():
    for wrong in [dict(LABEL, etki=True), dict(LABEL, etiketler=['DIGER', 'EMEK']), dict(LABEL, etiketler=['EMEK', 'EMEK']), dict(LABEL, freshness=1), dict(LABEL, aciliyet=9)]:
        with pytest.raises(ValueError):
            p.validate_label(wrong)


def test_label_resume_and_prompt_change(server, tmp_path):
    config, calls, _ = server
    source, out, cfg, prompt = [tmp_path / x for x in ['articles.jsonl', 'labels.jsonl', 'config.json', 'prompt.txt']]
    p.write_rows(source, [article('A')]); cfg.write_text(json.dumps(config)); prompt.write_text('rules')
    args = argparse.Namespace(input=source, out=out, config=cfg, prompt=prompt, log=None, retries=0, limit=0)
    p.label(args); p.label(args)
    assert len(calls) == 1 and len(p.rows(out)) == 1
    config['temperature'] = 0.7; cfg.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        p.label(args)
    config.pop('temperature'); cfg.write_text(json.dumps(config))
    prompt.write_text('new rules')
    with pytest.raises(ValueError):
        p.label(args)


def test_deterministic_grouped_splits_and_headline_only_messages(tmp_path):
    values = [{**article(f'Başlık {i}'), 'analysis': LABEL, 'teacher_signature': 'same', 'event_group': str(i // 2)} for i in range(200)]
    source = tmp_path / 'labels.jsonl'; prompt = tmp_path / 'prompt.txt'; out = tmp_path / 'prepared'
    p.write_rows(source, values); prompt.write_text('short')
    args = argparse.Namespace(input=source, student_prompt=prompt, reviewed_only=False, seed=42, out=out)
    p.prepare(args)
    first = {split: p.rows(out / (split + '.jsonl')) for split in ['train', 'validation', 'test']}
    training.check_splits(first)
    assert sum(map(len, first.values())) == 200
    assert all(first.values())
    for group in first.values():
        for row in group:
            assert row['messages'][1]['content'] == row['title']
            assert 'published' not in p.compact(row['messages'])
    p.prepare(args)
    assert first == {split: p.rows(out / (split + '.jsonl')) for split in first}
    first['test'].append(first['train'][0])
    with pytest.raises(ValueError):
        training.check_splits(first)


class FakeTokenizer:
    eos_token_id = 999
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        assert kwargs['return_dict'] is False
        assert [m['role'] for m in messages] == ['system', 'user']
        return [1001, 1002, 1003]
    def __call__(self, text, **kwargs):
        return {'input_ids': list(text.encode('utf-8'))}


def test_only_assistant_json_and_eos_receive_loss_and_long_rows_are_dropped():
    row = {'title': 'Başlık', 'messages': [{'role': 'system', 'content': 'short'}, {'role': 'user', 'content': 'Başlık'}, {'role': 'assistant', 'content': p.compact(LABEL)}]}
    encoded = training.encode_example(FakeTokenizer(), row, 512)
    assert encoded['labels'][:3] == [-100] * 3
    assert encoded['labels'][3:] == list(p.compact(LABEL).encode()) + [999]
    assert encoded['input_ids'][3:] == encoded['labels'][3:]
    assert training.encode_example(FakeTokenizer(), row, 4) is None
    row['messages'][1]['content'] = 'metadata plus title'
    with pytest.raises(ValueError):
        training.encode_example(FakeTokenizer(), row, 512)


def test_invalid_evaluation_is_not_dropped_from_denominator(server, tmp_path):
    config, calls, mode = server
    mode['finish'] = 'length'
    cfg = tmp_path / 'config.json'; cfg.write_text(json.dumps(config))
    data = tmp_path / 'test.jsonl'
    p.write_rows(data, [{'id': 'a', 'title': 'A', 'messages': [{'role': 'system', 'content': 'short'}, {'role': 'user', 'content': 'A'}, {'role': 'assistant', 'content': p.compact(LABEL)}]}])
    out = tmp_path / 'report.json'
    p.evaluate(argparse.Namespace(config=cfg, input=data, out=out))
    report = json.loads(out.read_text())
    assert report['invalid'] == 1 and report['topic_micro_f1'] == 0 and report['score_mae']['etki'] == 3


@pytest.fixture
def benchmark_setup(tmp_path):
    source, prompt, data = tmp_path / 'labels.jsonl', tmp_path / 'prompt.txt', tmp_path / 'prepared'
    p.write_rows(source, [{**article(f'Benchmark {i}'), 'analysis': LABEL,
                           'teacher_signature': 'same', 'event_group': str(i // 2)} for i in range(200)])
    prompt.write_text('short student prompt')
    p.prepare(argparse.Namespace(input=source, student_prompt=prompt, reviewed_only=False, seed=42, out=data))
    manifest = tmp_path / 'distill-manifest.json'
    manifest.write_text(json.dumps({'dataset_manifest': json.loads((data / 'manifest.json').read_text())}))
    old, new = tmp_path / 'old.json', tmp_path / 'new.json'
    old.write_text(json.dumps({'model': 'old'})); new.write_text(json.dumps({'model': 'new'}))
    return argparse.Namespace(data=data, new_manifest=manifest, old_config=old, new_config=new,
                              warmup=1, out=tmp_path / 'benchmark.json')


def test_benchmark_pairs_only_held_out_examples_and_penalizes_invalid(benchmark_setup, monkeypatch):
    args = benchmark_setup
    test = p.rows(args.data / 'test.jsonl')
    allowed = {row['title'] for row in test}
    seen = {'old': [], 'new': []}
    class Client:
        def __init__(self, config):
            self.model = config['model']
        def classify(self, title, prompt):
            assert title in allowed and prompt == 'short student prompt'
            seen[self.model].append(title)
            if self.model == 'new':
                raise ValueError('invalid response')
            return LABEL.copy(), {}
    monkeypatch.setattr(p, 'ChatClient', Client)
    args.warmup = 0
    p.benchmark(args)
    report = json.loads(args.out.read_text())
    n = len(test)
    assert seen['old'] == seen['new'] and len(seen['old']) == n
    assert report['old']['exact_match'] == 1 and report['new']['exact_match'] == 0
    assert report['new']['invalid'] == n and report['new']['score_mae']['etki'] == 3
    assert report['delta_new_minus_old']['exact_match'] == -1
    assert report['delta_new_minus_old']['invalid'] == n
    assert report['old']['latency_seconds']['mean'] >= 0
    pairs = p.rows(str(args.out) + '.predictions.jsonl')
    assert len(pairs) == n and pairs[0]['old']['prediction'] == LABEL
    assert pairs[0]['truth'] == LABEL and pairs[0]['new']['error'] == 'ValueError'


def test_benchmark_excludes_warmup(benchmark_setup, monkeypatch):
    args = benchmark_setup
    seen = []
    class Client:
        def __init__(self, config):
            self.model = config['model']
        def classify(self, title, prompt):
            seen.append((self.model, title))
            return LABEL.copy(), {}
    monkeypatch.setattr(p, 'ChatClient', Client)
    p.benchmark(args)
    n = len(p.rows(args.data / 'test.jsonl'))
    assert len(seen) == 2 * (n + 1)
    assert json.loads(args.out.read_text())['count'] == n


def test_benchmark_releases_each_model_before_loading_next(benchmark_setup):
    events = []
    class Client:
        def __init__(self, config):
            self.model = config['model']
            events.append(('load', self.model))
        def classify(self, title, prompt):
            return LABEL.copy(), {}
        def close(self):
            events.append(('unload', self.model))
    p.benchmark(benchmark_setup, client_factory=Client)
    assert events == [('load', 'old'), ('unload', 'old'), ('load', 'new'), ('unload', 'new')]


def test_benchmark_releases_model_when_warmup_fails(benchmark_setup):
    closed = []
    class Client:
        def __init__(self, config):
            self.model = config['model']
        def classify(self, title, prompt):
            raise ConnectionError('unavailable')
        def close(self):
            closed.append(self.model)
    with pytest.raises(ConnectionError):
        p.benchmark(benchmark_setup, client_factory=Client)
    assert closed == ['old']


@pytest.mark.parametrize('problem', ['hash', 'manifest', 'settings', 'leak', 'empty'])
def test_benchmark_rejects_unfair_or_modified_data_before_api(benchmark_setup, monkeypatch, problem):
    args = benchmark_setup
    manifest = json.loads((args.data / 'manifest.json').read_text())
    if problem == 'hash':
        with (args.data / 'test.jsonl').open('a') as f:
            f.write('\n')
    elif problem == 'manifest':
        args.new_manifest.write_text(json.dumps({'dataset_manifest': {}}))
    elif problem == 'settings':
        args.new_config.write_text(json.dumps({'model': 'new', 'max_tokens': 1}))
    else:
        values = p.rows(args.data / 'test.jsonl')
        p.write_rows(args.data / 'test.jsonl', values + [p.rows(args.data / 'train.jsonl')[0]] if problem == 'leak' else [])
        manifest['split_sha256']['test'] = __import__('hashlib').sha256((args.data / 'test.jsonl').read_bytes()).hexdigest()
        (args.data / 'manifest.json').write_text(json.dumps(manifest))
        args.new_manifest.write_text(json.dumps({'dataset_manifest': manifest}))
    def no_api(config):
        pytest.fail('Invalid benchmark must not call API')
    monkeypatch.setattr(p, 'ChatClient', no_api)
    with pytest.raises(ValueError):
        p.benchmark(args)
    assert not args.out.exists()

@pytest.mark.parametrize('valid', [True, False])
def test_export_commands_verify_gguf_and_publish_hash_only_on_success(tmp_path, monkeypatch, valid):
    import export_gguf as export
    model = tmp_path / 'model'; model.mkdir(); (model / 'config.json').write_text('{}')
    llama = tmp_path / 'llama'; llama.mkdir(); (llama / 'convert_hf_to_gguf.py').write_text('# stub')
    quantizer = tmp_path / 'quantize'; quantizer.write_text('# stub')
    out = tmp_path / 'export' / 'student.gguf'
    calls = []
    def run(command, check):
        assert check is True
        calls.append(command)
        if '--outfile' in command:
            assert command[-2:] == ['--outtype', 'f16']
            Path(command[command.index('--outfile') + 1]).write_bytes(b'GGUFf16')
        else:
            assert command[-1] == 'Q4_K_M'
            Path(command[2]).write_bytes(b'GGUFquant' if valid else b'bad')
    monkeypatch.setattr(export.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['export_gguf.py', '--model', str(model), '--llama-dir', str(llama), '--quantizer', str(quantizer), '--out', str(out)])
    if valid:
        export.main()
        metadata = json.loads(out.with_suffix('.gguf.json').read_text())
        assert metadata['sha256'] == __import__('hashlib').sha256(out.read_bytes()).hexdigest()
        assert len(calls) == 2
        with pytest.raises(SystemExit):
            export.main()
    else:
        with pytest.raises(ValueError):
            export.main()
        assert not out.exists() and not out.with_suffix('.gguf.json').exists()


def test_api_redirect_is_not_followed(server):
    import urllib.error
    config,calls,mode=server
    mode['redirect']=True
    with pytest.raises(urllib.error.HTTPError) as error:
        p.ChatClient(config).classify('A','B')
    assert error.value.code==302 and len(calls)==1


@pytest.mark.parametrize('target,per_feed,expected', [(50, 0, 50), (0, 0, 62), (100, 0, 62), (100, 2, 3), (1, 0, 1)])
def test_collect_target_resume_duplicates_and_feed_limits(tmp_path, monkeypatch, target, per_feed, expected):
    feeds, out, logfile = [tmp_path / x for x in ('feeds.json', 'headlines.jsonl', 'collection.jsonl')]
    feeds.write_text(json.dumps([{'id': 'a', 'url': 'https://example.org/a'}, {'id': 'b', 'url': 'https://example.org/b'}]))
    original = article('Existing')
    p.write_rows(out, [original])
    calls = []
    def fetch(url):
        calls.append(url)
        titles = ['Existing', 'Existing'] + [f'Headline {i}' for i in range(60)] if url.endswith('/a') else ['Headline 0', 'Last']
        return ('<rss><channel>' + ''.join(f'<item><title>{title}</title></item>' for title in titles) + '</channel></rss>').encode()
    monkeypatch.setattr(p, 'get_url', fetch)
    p.collect(argparse.Namespace(feeds=feeds, out=out, log=logfile, count=target, per_feed=per_feed))
    result = p.rows(out)
    assert len(result) == expected
    assert next(row for row in result if row['id'] == original['id']) == original
    events = p.rows(logfile)
    assert events[-1]['added'] == expected - 1
    assert any(e['event'] == 'collection_target_not_reached' for e in events) == (target > expected)
    if target == 1:
        assert calls == []
    elif target == 50:
        assert len(calls) == 1


@pytest.mark.parametrize('option', ['--count', '--per-feed'])
def test_collect_rejects_negative_counts(monkeypatch, option):
    monkeypatch.setattr(sys, 'argv', ['pipeline.py', 'collect', '--out', 'unused.jsonl', option, '-1'])
    with pytest.raises(SystemExit) as error:
        p.main()
    assert error.value.code == 2


def test_label_sends_fill_template_and_schema_but_no_metadata(server, tmp_path):
    config,calls,_=server
    source,out,cfg,prompt=[tmp_path/x for x in ['input.jsonl','labels.jsonl','teacher.json','teacher.txt']]
    p.write_rows(source,[article('Başlık')]);cfg.write_text(json.dumps(config));prompt.write_text('rules')
    p.label(argparse.Namespace(input=source,out=out,config=cfg,prompt=prompt,log=None,retries=0,limit=0,progress='plain'))
    request=calls[0]
    assert request['messages'][1]=={'role':'user','content':'Başlık'}
    assert '"tur": null' in request['messages'][0]['content']
    assert request['response_format']['json_schema']['strict'] is True
    assert set(request['response_format']['json_schema']['schema']['required'])==set(p.FIELDS)
    assert p.rows(out)[0]['teacher_system_sha256']==p.digest(p.teacher_system('rules'))


def test_progress_never_advances_for_inflight_or_failed_labels():
    import io
    from progress import TeacherProgress
    stream=io.StringIO()
    with TeacherProgress(10,3,mode='plain',stream=stream) as progress:
        progress.begin('abc');assert progress.valid==3
        progress.failure();assert progress.valid==3 and progress.errors==1
        progress.begin('next');progress.success();assert progress.valid==4
    text=stream.getvalue()
    assert '4/10' in text and 'kalan 6' in text and 'hata 1' in text
    assert '%100.0' not in text


def test_legacy_valid_labels_are_reused_with_new_output_template(server,tmp_path):
    config,calls,_=server
    source,out,cfg,prompt=[tmp_path/x for x in ['input.jsonl','labels.jsonl','teacher.json','teacher.txt']]
    a=article('A');p.write_rows(source,[a]);cfg.write_text(json.dumps(config));prompt.write_text('rules')
    old={**a,'analysis':LABEL,'teacher_signature':p.teacher_signature(config,'rules')}
    p.write_rows(out,[old])
    p.label(argparse.Namespace(input=source,out=out,config=cfg,prompt=prompt,log=None,retries=0,limit=0,progress='plain'))
    assert not calls and p.rows(out)==[old]


def test_launcher_label_only_runs_from_any_directory_and_resumes(server,tmp_path,monkeypatch):
    import start_distillation as starter
    config,calls,_=server
    source=tmp_path/'headlines.jsonl';p.write_rows(source,[article('A')])
    cfg=tmp_path/'teacher.json';cfg.write_text(json.dumps(config))
    args=['start_distillation.py','--work-dir',str(tmp_path/'work dir'),'--headlines',str(source),
          '--config',str(cfg),'--label-only','--progress','plain']
    monkeypatch.setattr(sys,'argv',args)
    assert starter.main()==0
    before=len(calls)
    assert starter.main()==0 and len(calls)==before
    assert len(p.rows(tmp_path/'work dir'/'labels.jsonl'))==1


def test_unfilled_template_is_not_saved_or_counted_successful(server,tmp_path,capsys):
    config,calls,mode=server
    mode['content']=json.dumps(p.OUTPUT_TEMPLATE)
    source,out,cfg,prompt=[tmp_path/x for x in ['input.jsonl','labels.jsonl','config.json','prompt.txt']]
    p.write_rows(source,[article('A')]);cfg.write_text(json.dumps(config));prompt.write_text('rules')
    with pytest.raises(ValueError):
        p.label(argparse.Namespace(input=source,out=out,config=cfg,prompt=prompt,log=None,retries=0,limit=0,progress='plain'))
    assert not p.rows(out)
    text=capsys.readouterr().err
    assert '0/1' in text and 'hata 1' in text and '%100.0' not in text


def test_progress_heartbeat_runs_during_request_and_stops_cleanly():
    import io
    from progress import TeacherProgress
    signaled=threading.Event()
    main_thread=threading.get_ident()
    class Capture(io.StringIO):
        def write(self,value):
            if threading.get_ident()!=main_thread:
                signaled.set()
            return super().write(value)
    with TeacherProgress(1,mode='bar',stream=Capture()) as progress:
        progress.begin('id')
        assert signaled.wait(2)
        assert progress.valid==0
    assert not progress.thread.is_alive()


def test_loading_stage_reports_completion_and_failure_without_fake_percentage():
    import io
    from progress import StageProgress
    stream = io.StringIO()
    with StageProgress('Model yükleniyor', 'plain', stream) as stage:
        pass
    assert not stage.thread.is_alive()
    assert 'başladı' in stream.getvalue() and 'tamamlandı' in stream.getvalue()
    assert '%' not in stream.getvalue()
    with pytest.raises(ValueError):
        with StageProgress('Model yükleniyor', 'plain', stream):
            raise ValueError('loader failed')
    assert 'durduruldu' in stream.getvalue()


def test_launcher_forwards_carriage_returns_before_child_exits(tmp_path, monkeypatch):
    import io
    import os
    import start_distillation as starter
    marker = tmp_path / 'visible'
    captured = io.StringIO()
    class Capture:
        def write(self, text):
            captured.write(text)
            if '\rMODEL 1/2' in captured.getvalue():
                marker.touch()
        def flush(self):
            pass
    monkeypatch.setattr(sys, 'stdout', Capture())
    code = ('import os,time\n'
            'os.write(1, "\\rMODEL 1/2".encode())\n'
            f'path={str(marker)!r}\n'
            'for i in range(100):\n'
            ' if os.path.exists(path): break\n'
            ' time.sleep(0.02)\n'
            'else: raise RuntimeError("progress was buffered")\n'
            'data="\\rMODEL 2/2 İndirildi\\n".encode()\n'
            'for byte in data: os.write(1, bytes([byte]))\n')
    log = tmp_path / 'stage.log'
    starter.run_command([sys.executable, '-c', code], log)
    assert marker.exists()
    assert '\rMODEL 1/2' in captured.getvalue()
    assert 'İndirildi' in captured.getvalue() and '�' not in captured.getvalue()
    assert b'\rMODEL 1/2' in log.read_bytes()


def test_download_student_pins_revision_and_reuses_local_models(tmp_path, monkeypatch):
    import types
    calls = []
    fake_hub = types.SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs) or '/cached/snapshot')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_hub)
    monkeypatch.setattr(training, 'download_tqdm', lambda mode: 'test-tqdm-' + mode)
    assert training.download_student(str(tmp_path), None, 'bar') == str(tmp_path)
    assert not calls
    assert training.download_student('LiquidAI/LFM3-1.7B', 'fixed-commit', 'bar') == '/cached/snapshot'
    assert calls[0]['revision'] == 'fixed-commit'
    assert calls[0]['tqdm_class'] == 'test-tqdm-bar'
    assert '*.safetensors' in calls[0]['allow_patterns']
    assert calls[0]['max_workers'] == 2


def test_hub_download_uses_https_defaults_but_preserves_explicit_overrides(monkeypatch):
    for key in ('HF_HUB_DISABLE_XET', 'HF_HUB_DOWNLOAD_TIMEOUT', 'HF_HUB_ETAG_TIMEOUT'):
        monkeypatch.delenv(key, raising=False)
    training.configure_hub_download()
    assert training.os.environ['HF_HUB_DISABLE_XET'] == '1'
    assert training.os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] == '60'
    monkeypatch.setenv('HF_HUB_DISABLE_XET', '0')
    monkeypatch.setenv('HF_HUB_DOWNLOAD_TIMEOUT', '120')
    training.configure_hub_download()
    assert training.os.environ['HF_HUB_DISABLE_XET'] == '0'
    assert training.os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] == '120'


def test_download_progress_visible_through_pipe_and_off_is_quiet(monkeypatch):
    import io
    from progress import download_tqdm
    stream = io.StringIO()
    monkeypatch.setattr(sys, 'stderr', stream)
    with download_tqdm('bar')(total=100, desc='İndirme', unit='B', miniters=1) as progress:
        assert progress.disable is False
        progress.update(50)
        progress.refresh()
    assert '50%' in stream.getvalue()
    stream.seek(0); stream.truncate()
    with download_tqdm('off')(total=100) as progress:
        progress.update(100)
    assert stream.getvalue() == ''


def test_empty_label_input_does_not_call_teacher(server,tmp_path):
    config,calls,_=server
    source,out,cfg,prompt=[tmp_path/x for x in ['input.jsonl','labels.jsonl','config.json','prompt.txt']]
    p.write_rows(source,[]);cfg.write_text(json.dumps(config));prompt.write_text('rules')
    with pytest.raises(ValueError):
        p.label(argparse.Namespace(input=source,out=out,config=cfg,prompt=prompt,log=None,retries=0,limit=0,progress='plain'))
    assert not calls


def test_progress_bar_fits_standard_terminal(monkeypatch):
    import progress as ui
    import os
    monkeypatch.setattr(ui.shutil,'get_terminal_size',lambda **kwargs:os.terminal_size((80,24)))
    display=ui.TeacherProgress(1000,420,mode='bar')
    display.status='işleniyor abcdef01'
    text=display.line()
    assert len(text)<=79 and '420/1000' in text and 'kalan 580' in text and 'hata 0' in text


def test_parallel_labels_limit_failure_and_resume(tmp_path, monkeypatch):
    source, out, cfg, prompt = [tmp_path / name for name in ('input.jsonl', 'labels.jsonl', 'config.json', 'prompt.txt')]
    p.write_rows(source, [article(f'Parallel {i}') for i in range(8)])
    cfg.write_text(json.dumps({'model': 'test', 'base_url': 'http://localhost/v1'}))
    prompt.write_text('rules')
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    state = {'active': 0, 'peak': 0, 'calls': 0, 'fail': True}
    def classify(self, title, system):
        with lock:
            state['active'] += 1
            state['calls'] += 1
            state['peak'] = max(state['peak'], state['active'])
        if state['fail']:
            barrier.wait(timeout=5)
        with lock:
            state['active'] -= 1
        if state['fail'] and title == fail_title:
            raise ValueError('invalid output')
        return LABEL, {'completion_tokens': 50}
    fail_title = sorted(p.rows(source), key=lambda r: (r['published'], r['id']), reverse=True)[0]['title']
    monkeypatch.setattr(p.ChatClient, 'classify', classify)
    args = argparse.Namespace(input=source, out=out, config=cfg, prompt=prompt, log=None, retries=0, limit=4, workers=4, progress='off')
    with pytest.raises(ValueError, match='1 labels failed'):
        p.label(args)
    assert state['peak'] == 4 and state['calls'] == 4
    assert len(p.rows(out)) == 3
    state['fail'] = False
    args.limit = 0
    p.label(args)
    assert len(p.rows(out)) == 8
    assert len({r['id'] for r in p.rows(out)}) == 8
    assert state['calls'] == 9
    assert len([r for r in p.rows(str(out) + '.log.jsonl') if r['event'] == 'teacher_completed']) == 8


def test_q_key_restores_terminal_and_requests_exit():
    import os
    import pty
    import termios
    from controls import StopControl
    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    try:
        with os.fdopen(os.dup(slave), 'r') as stream:
            with StopControl(stream) as control:
                assert not control.requested.is_set()
                os.write(master, b'q')
                assert control.requested.wait(2)
                assert control.stop_file.exists()
            assert termios.tcgetattr(slave) == original
            assert not control.thread.is_alive()
    finally:
        os.close(master)
        os.close(slave)


def test_save_exit_drains_active_labels_and_resumes(tmp_path, monkeypatch):
    from controls import SavedExit, stop_session
    source, out, cfg, prompt = [tmp_path / name for name in ('input.jsonl', 'labels.jsonl', 'config.json', 'prompt.txt')]
    p.write_rows(source, [article(f'Stop {i}') for i in range(6)])
    cfg.write_text(json.dumps({'model': 'test', 'base_url': 'http://localhost/v1'}))
    prompt.write_text('rules')
    calls = []
    with stop_session() as control:
        def classify(self, title, system):
            calls.append(title)
            control.request()
            return LABEL, {}
        monkeypatch.setattr(p.ChatClient, 'classify', classify)
        args = argparse.Namespace(input=source, out=out, config=cfg, prompt=prompt,
                                  log=None, retries=0, limit=0, workers=1, progress='off')
        with pytest.raises(SavedExit):
            p.label(args)
    assert len(calls) == 1 and len(p.rows(out)) == 1
    monkeypatch.setattr(p.ChatClient, 'classify', lambda *args: (LABEL, {}))
    p.label(args)
    assert len(p.rows(out)) == 6


@pytest.mark.parametrize('outcome,shutdowns,expected', [('success', 1, 0), ('stop', 0, 130), ('error', 0, 1)])
def test_shutdown_only_after_success(monkeypatch, outcome, shutdowns, expected):
    import start_distillation as starter
    from controls import SavedExit
    calls = []
    def workflow(args):
        if outcome == 'stop':
            raise SavedExit()
        if outcome == 'error':
            raise RuntimeError('failed')
    monkeypatch.setattr(starter, 'workflow', workflow)
    monkeypatch.setattr(starter.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(sys, 'argv', ['start_distillation.py', '--label-only', '--shutdown'])
    assert starter.main() == expected
    assert calls == [['shutdown', '-h', '+1']] * shutdowns


def test_training_stop_requests_checkpoint_at_step_boundary(tmp_path, monkeypatch):
    from types import SimpleNamespace
    stop = tmp_path/'stop'
    monkeypatch.setenv('DISTILL_STOP_FILE', str(stop))
    control = SimpleNamespace(should_save=False, should_training_stop=False, should_evaluate=False)
    training.save_and_stop(None, None, control)
    assert not control.should_save and not control.should_training_stop
    stop.touch()
    training.save_and_stop(None, None, control)
    assert control.should_save and control.should_training_stop and control.should_evaluate
