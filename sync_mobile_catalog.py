#!/usr/bin/env python3
"""Build the 20-entry app catalog from measured scores and verified upload receipts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT.parent.parent


def write_docs(rows):
    (ROOT / 'docs/mobile-catalog.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    for tr in (False, True):
        title = '# Mobil modeller' if tr else '# Mobile models'
        intro = ('Gündem Android model ekranında 10 ailenin hem orijinal hem fine-tuned sürümünü seçebilirsiniz. '
                 'Her kartta Gündem Score gösterilir. Aşağıdaki dosyalar tam modeldir; ayrıca adapter uygulamak gerekmez.' if tr else
                 'The Gündem Android model screen offers original and fine-tuned versions of all 10 families, '
                 'with Gündem Score on each card. These are standalone full models; no adapter application is required.')
        note = ('Puanlar 101 test başlığında HF/BF16 ile ölçüldü; mobil dosyalar yeniden puanlanmadı. '
                '0,0 ölçülmüş sonuçtur: orijinal modeller bu kısıtlanmamış kıyasta geçerli JSON üretemedi. '
                'Uygulama JSON kısıtlaması kullanır. Puanlar telefon hızını veya genel zekâyı ölçmez.' if tr else
                'Scores were measured on 101 test headlines using HF/BF16; mobile files were not rescored. '
                '0.0 is a measured result: the original models failed to produce valid JSON in that unconstrained benchmark. '
                'The app uses constrained JSON. These scores do not measure phone speed or general intelligence.')
        text = title + '\n\n' + intro + '\n\n[Android APK](https://github.com/efebaykaraa/gundem-android/releases/tag/v0.9.1) · '
        text += '[JSON catalog](mobile-catalog.json) · [Export instructions](MOBILE-EXPORT.md)\n\n' + note + '\n\n'
        text += ('| Model | Sürüm | Gündem Score / 100 | MB | İndir |\n' if tr else
                 '| Model | Variant | Gündem Score / 100 | MB | Download |\n')
        text += '| --- | --- | ---: | ---: | --- |\n'
        for row in rows:
            variant = 'Fine-tuned' if row['variant'] == 'distilled' else ('Orijinal' if tr else 'Original')
            fmt = 'ONNX INT8 ZIP' if row['format'] == 'onnx' else 'Q4_K_M GGUF'
            score = f'{row["overallScore"]:.1f}'
            if tr:
                score = score.replace('.', ',')
            text += f'| {row["family"]} | {variant} | {score} | {row["size"]/1_000_000:.0f} | [{fmt}]({row["url"]}) |\n'
        text += ('\nİndirme bağlantıları sabit commit’e bağlıdır; boyut ve SHA-256 JSON kataloğunda bulunur. '
                 'Modellerin RAM ihtiyacı cihaz ve girdiye göre değişir. Her model kendi lisansını korur; '
                 'Octen ticari olmayan kullanımla sınırlıdır.\n' if tr else
                 '\nDownloads are pinned to commits; sizes and SHA-256 are in the JSON catalog. '
                 'RAM requirements depend on the device and input. Each model retains its own license; '
                 'Octen is restricted to non-commercial use.\n')
        (ROOT / ('docs/MOBILE.tr.md' if tr else 'docs/MOBILE.md')).write_text(text)


def main():
    reports = json.loads((ROOT / 'workspace/reports/leaderboard.json').read_text())['results']
    index = json.loads((ROOT / 'workspace/releases/models-v1.0.0/release-index.json').read_text())['models']
    receipts = json.loads((ROOT / 'workspace/releases/models-v1.0.0/hf-published.json').read_text())
    commits = {r['repo_id']: r['commit'] for r in receipts}
    urls = {}
    # Preserve the two already-published usable GGUFs while new pairs are exported.
    for e in index:
        if e['gguf']:
            metadata = json.loads((ROOT / f'workspace/mobile/{e["model_id"]}-distilled-Q4_K_M.json').read_text())
            urls[(e['model_id'], 'distilled')] = {**metadata,
                'url': f'https://huggingface.co/{e["repo_id"]}/resolve/{commits[e["repo_id"]]}/gguf/{e["gguf"]}'}
    for path in (ROOT / 'workspace/mobile/published').glob('*.json'):
        for row in json.loads(path.read_text()):
            urls[(row['model_id'], row['variant'])] = row
    expected = {(r['model_id'], r['variant']) for r in reports}
    missing = expected - urls.keys()
    if missing:
        raise SystemExit('Refusing incomplete catalog; publish these variants first: ' + str(sorted(missing)))
    assert len(expected) == 20
    families = list(dict.fromkeys(r['model_id'] for r in reports))
    families.remove('lfm3-base')
    families.insert(0, 'lfm3-base')
    rows = []
    for model_id in families:
        for variant in ('distilled', 'original'):
            report = next(r for r in reports if r['model_id'] == model_id and r['variant'] == variant)
            artifact = urls.get((model_id, variant))
            family = report['label']
            label = 'Fine-tuned' if variant == 'distilled' else 'Orijinal'
            name = ('Gündem ' if variant == 'distilled' and not model_id.startswith('llama-') else '') + family + ' · ' + label
            reason = ''
            rows.append({'id': model_id + ':' + variant, 'family': family, 'variant': variant,
                'name': name, 'overallScore': report['overall_score'], 'scoreVersion': report['overall_score_version'],
                'url': artifact['url'] if artifact else None, 'size': artifact['size'] if artifact else 0,
                'sha256': artifact['sha256'] if artifact else '',
                'format': artifact.get('format', 'gguf') if artifact else 'gguf',
                'ram': 'Bellek ihtiyacı cihaz ve bağlama göre değişir.', 'availabilityReason': reason})
    (APP / 'app/src/main/assets/models.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    # The public default API and the bundled catalog must never drift apart.
    first = rows[0]
    assert first['url'] and first['variant'] == 'distilled'
    quote = lambda value: json.dumps(value, ensure_ascii=False)
    arguments = [quote(first['url']), str(first['size']) + 'L', quote(first['sha256']), quote(first['name']),
                 quote(first['ram']), quote(first['id']), quote(first['family']), quote(first['variant']),
                 str(first['overallScore']), quote(first['scoreVersion'])]
    path = APP / 'app/src/main/java/tr/gundem/ModelDownload.kt'
    text = path.read_text()
    previous = next(line for line in text.splitlines() if line.startswith('val RECOMMENDED_MODEL='))
    path.write_text(text.replace(previous, 'val RECOMMENDED_MODEL=ModelSpec(' + ','.join(arguments) + ')'))
    url_parts = first['url'].split('/')
    recommended = APP / 'docs/recommended-model.json'
    details = json.loads(recommended.read_text())
    details.update(provider='/'.join(url_parts[3:5]), revision=url_parts[6],
                   file='/'.join(url_parts[7:]), size=first['size'], sha256=first['sha256'])
    recommended.write_text(json.dumps(details, indent=2) + '\n')
    write_docs(rows)
    print(f'Catalog: {len(rows)} variants; {sum(bool(r["url"]) for r in rows)} verified downloads.')


if __name__ == '__main__':
    main()
