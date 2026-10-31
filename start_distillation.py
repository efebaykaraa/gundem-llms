#!/usr/bin/env python3
"""One-command teacher labeling -> LFM training -> merge, with optional GGUF export."""
import argparse
import codecs
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
import venv

import pipeline as p
from controls import SavedExit, current_control, stop_session
from progress import progress_mode

HERE = Path(__file__).resolve().parent


def check_teacher(config):
    print('Öğretmen API kontrol ediliyor…', flush=True)
    url = config['base_url'].rstrip('/') + '/models'
    client = p.ChatClient(config)  # Validate URL and disable API redirects.
    headers = client.auth_headers()
    try:
        with client.opener.open(urllib.request.Request(url, headers=headers), timeout=15) as response:
            result = json.loads(response.read(2 * 1024 * 1024))
    except urllib.error.HTTPError as error:
        if error.code in (404, 405):
            print('Hizmet /models sunmuyor; model ilk etiketleme çağrısında kontrol edilecek.', flush=True)
            return
        raise RuntimeError(f'Öğretmen API kontrolü başarısız: HTTP {error.code}') from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError('Öğretmen API erişilemiyor. Bağlantıyı veya --config dosyasındaki base_url adresini kontrol edin.') from None
    ids = {row.get('id') for row in result.get('data', [])}
    if config['model'] not in ids:
        raise RuntimeError('Öğretmen model kimliği /models listesinde yok; config içindeki model adını hizmetinizle eşleştirin.')
    print('Öğretmen hazır: ' + config['model'], flush=True)


def run_command(command, log):
    control = current_control()
    if control:
        control.check()
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    environment.setdefault('HF_HUB_DISABLE_XET', '1')
    environment.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
    environment.setdefault('HF_HUB_ETAG_TIMEOUT', '30')
    if control:
        environment['DISTILL_STOP_FILE'] = str(control.stop_file)
    print('Çalışıyor: ' + ' '.join(str(x) for x in command[:3]), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as output:
        process = subprocess.Popen([str(x) for x in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=environment,
                                   bufsize=0)
        try:
            decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
            while True:
                chunk = os.read(process.stdout.fileno(), 8192)
                content = decoder.decode(chunk, final=not chunk)
                if content:
                    print(content, end='', flush=True)
                    output.write(content); output.flush()
                if not chunk:
                    break
            code = process.wait()
            if control and code in (0, 130):
                control.check()
            if code:
                raise RuntimeError(f'Aşama başarısız (kod {code}). Log: {log}')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            raise


def training_python(args, work):
    print('Eğitim ortamı kontrol ediliyor…', flush=True)
    python = args.training_python
    if args.install:
        if python:
            raise ValueError('--install ile --training-python birlikte kullanılamaz')
        environment = work / 'venv-train'
        if not (environment / 'bin/python').is_file():
            print('İzole eğitim ortamı kuruluyor…', flush=True)
            venv.EnvBuilder(with_pip=True).create(environment)
        python = str(environment / 'bin/python')
        run_command([python, '-m', 'pip', 'install', '-r', HERE/'requirements-train.txt'], work/'setup.log')
    elif not python:
        candidates = [work/'venv-train/bin/python', HERE/'.venv-train/bin/python']
        python = str(next((x for x in candidates if x.is_file()), Path(sys.executable)))
    result = subprocess.run([python, '-c', 'import torch,transformers,peft; print("cuda="+str(torch.cuda.is_available()))'],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Eğitim bağımlılıkları eksik. --install veya --training-python /venv/bin/python kullanın. Yalnız etiketleme için --label-only.')
    if not args.cpu and 'cuda=True' not in result.stdout:
        raise RuntimeError('CUDA GPU bulunamadı. GPU ortamı seçin; sadece etiketlemek için --label-only, bilinçli CPU eğitimi için --cpu kullanın.')
    return python


def checkpoint(adapter):
    choices = []
    for path in adapter.glob('checkpoint-*'):
        number = path.name.removeprefix('checkpoint-')
        if number.isdigit() and (path/'trainer_state.json').is_file():
            choices.append((int(number), path))
    return max(choices, default=(0, None))[1]


def workflow(args):
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work/'.start.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Bu çalışma klasöründe başka bir damıtma işlemi çalışıyor.') from None
        adapter, merged = work/'adapter', work/'merged'
        if args.refresh and adapter.exists() and any(adapter.iterdir()):
            raise ValueError('Eğitim klasörü varken veri yenilenemez; yeni --work-dir seçin.')
        headlines = work/'headlines.jsonl'
        if args.headlines:
            source = Path(args.headlines).resolve()
            if not source.is_file():
                raise ValueError('Başlık dosyası bulunamadı')
            if headlines.exists() and headlines.read_bytes() != source.read_bytes():
                raise ValueError('Bu klasörde farklı başlık verisi var; yeni --work-dir seçin.')
            if source != headlines:
                shutil.copy2(source, headlines)
        elif args.refresh or not headlines.is_file():
            p.collect(argparse.Namespace(feeds=args.feeds, out=headlines, per_feed=0, count=args.count, log=work/'collection.log.jsonl'))
        config = p.load_teacher_config(args)
        prompt = Path(args.prompt).read_text(encoding='utf-8')
        signature = p.teacher_signature(config, prompt)
        previous = p.rows(work/'labels.jsonl')
        if any(r.get('teacher_signature') != signature for r in previous):
            raise ValueError('Bu klasör farklı öğretmen/talimatla kullanılmış; yeni --work-dir seçin.')
        for row in previous:
            p.validate_label(row['analysis'])
        articles = p.rows(headlines)
        if not articles:
            raise ValueError('Başlık dosyası boş; önce haber toplayın.')
        if any(r['id'] != p.title_id(r['title']) for r in articles):
            raise ValueError('Başlık/kimlik eşleşmiyor; veri dosyasını kontrol edin.')
        pending = {r['id'] for r in articles} - {r['id'] for r in previous}
        if pending:
            check_teacher(config)
        else:
            print('Etiketler hazır; öğretmene tekrar istek yapılmayacak.', flush=True)
        python = None if args.label_only else training_python(args, work)
        p.label(argparse.Namespace(input=headlines, out=work/'labels.jsonl', _teacher_config=config, prompt=args.prompt,
                                  retries=2, limit=0, log=work/'teacher.log.jsonl', progress=args.progress, workers=args.workers))
        if args.label_only:
            print(f'Etiketleme bitti: {work / "labels.jsonl"}', flush=True)
            return
        data = work/'prepared'
        p.prepare(argparse.Namespace(input=work/'labels.jsonl', out=data, student_prompt=HERE/'student-system.txt',
                                    reviewed_only=args.reviewed_only, seed=42))
        manifest = json.loads((data/'manifest.json').read_text())
        counts = manifest['counts']
        print(f'Veri ayrımı: eğitim={counts["train"]}, doğrulama={counts["validation"]}, '
              f'benchmark için ayrılan test={counts["test"]}. Test başlıkları eğitime girmez.', flush=True)
        result_manifest = adapter/'distill-manifest.json'
        if result_manifest.is_file() and (adapter/'adapter_model.safetensors').is_file():
            previous_run = json.loads(result_manifest.read_text())
            if (previous_run['dataset_manifest'] != manifest or previous_run['student'] !=
                (str(Path(args.student).resolve()) if Path(args.student).is_dir() else args.student) or
                previous_run['hyperparameters']['epochs'] != args.epochs or
                previous_run['hyperparameters']['max_steps'] != args.max_steps):
                raise ValueError('Eğitim verisi/model/ayar değişmiş; yeni --work-dir seçin.')
            if args.revision != 'main' and previous_run.get('revision') != args.revision:
                raise ValueError('Model sürümü değişmiş; yeni --work-dir seçin.')
            print('Eğitim zaten tamamlanmış; tekrar çalıştırılmıyor.', flush=True)
        else:
            resume = checkpoint(adapter)
            if adapter.exists() and any(adapter.iterdir()) and not resume:
                if not (adapter/'training-plan.json').is_file():
                    raise ValueError('Tanımlanamayan eğitim dosyaları var; yeni --work-dir seçin.')
                backup = work/('adapter-incomplete-' + dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
                adapter.rename(backup)
                print(f'Checkpoint yok; yarım çıktılar {backup} içinde korundu. Eğitim baştan başlıyor.', flush=True)
            command = [python, HERE/'train.py', 'train', '--data', data, '--out', adapter,
                       '--student', args.student, '--revision', args.revision, '--epochs', str(args.epochs),
                       '--max-steps', str(args.max_steps), '--progress', progress_mode(args.progress)]
            if resume:
                command += ['--resume', resume]
            if args.cpu:
                command += ['--cpu']
            if args.allow_small:
                command += ['--allow-small']
            run_command(command, work/'training.log')
        if (merged/'distill-manifest.json').is_file() and (merged/'config.json').is_file():
            if json.loads((merged/'distill-manifest.json').read_text()) != json.loads(result_manifest.read_text()):
                raise ValueError('Birleştirilmiş model farklı eğitime ait; yeni --work-dir seçin.')
            print('Birleştirilmiş model hazır.', flush=True)
        else:
            run_command([python, HERE/'train.py', 'merge', '--adapter', adapter, '--out', merged, '--cpu'], work/'merge.log')
        if args.gguf:
            target = work/'gundem-lfm-news-Q4_K_M.gguf'
            if target.exists():
                raise ValueError('GGUF çıktısı zaten var; yeniden üzerine yazılmadı.')
            run_command([args.convert_python or python, HERE/'export_gguf.py', '--model', merged,
                         '--llama-dir', args.llama_dir, '--quantizer', args.quantizer, '--out', target], work/'export.log')
        print(f'TAMAMLANDI. Model: {merged}\nLoglar ve veri: {work}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', default=HERE/'data')
    p.teacher_options(parser)
    parser.add_argument('--prompt', default=HERE/'teacher-system.txt')
    parser.add_argument('--feeds', default=HERE/'feeds.json'); parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--headlines'); parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--workers', type=int, default=16, help='Eşzamanlı öğretmen isteği (varsayılan: 16)')
    parser.add_argument('--progress', choices=['auto','bar','plain','off'], default='auto')
    parser.add_argument('--shutdown', action='store_true', help='Başarıyla bitince bilgisayarı 1 dakika sonra kapat')
    parser.add_argument('--label-only', action='store_true'); parser.add_argument('--reviewed-only', action='store_true')
    parser.add_argument('--install', action='store_true'); parser.add_argument('--training-python')
    parser.add_argument('--student', default='LiquidAI/LFM3-1.7B'); parser.add_argument('--revision', default='main')
    parser.add_argument('--epochs', type=float, default=3); parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--cpu', action='store_true'); parser.add_argument('--allow-small', action='store_true')
    parser.add_argument('--gguf', action='store_true'); parser.add_argument('--llama-dir'); parser.add_argument('--quantizer'); parser.add_argument('--convert-python')
    args = parser.parse_args()
    if args.workers < 1 or args.count < 0 or args.epochs <= 0 or args.max_steps == 0 or args.max_steps < -1:
        parser.error('Geçersiz sayı/epoch/adım değeri')
    if args.gguf and (not args.llama_dir or not args.quantizer):
        parser.error('--gguf için --llama-dir ve --quantizer gerekir')
    if args.label_only and args.gguf:
        parser.error('--label-only ile --gguf birlikte kullanılamaz')
    try:
        with stop_session() as control:
            workflow(args)
            control.check()
        if args.shutdown:
            subprocess.run(['shutdown', '-h', '+1'], check=True)
            print('Bilgisayar 1 dakika sonra kapanacak. İptal: shutdown -c', flush=True)
    except SavedExit:
        print('\nKaydedildi ve çıkıldı. Aynı komutla devam edebilirsiniz. Bilgisayar kapatılmayacak.', file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        print('\nDurduruldu; kaydedilmiş etiketler ve mevcut checkpointler korunur.', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'HATA: {error}', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
