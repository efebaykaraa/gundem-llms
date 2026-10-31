"""Aggregate only compatible completed results; export CSV, JSON, Markdown and plots."""
import csv
from pathlib import Path
from .common import read_json, write_json
from .scoring import score_report


METRICS = ['exact_match', 'topic_micro_f1', 'type_accuracy', 'invalid']


def aggregate(config, root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results, statuses, pairs = [], [], []
    reference = None
    for spec in config['models']:
        directory = root / 'experiments' / spec['id']
        state = read_json(directory / 'state.json') if (directory / 'state.json').exists() else {}
        statuses.append({'id': spec['id'], 'label': spec['label'], 'status': state.get('status', 'pending'),
                         'detail': state.get('detail', ''), 'note': spec.get('note', '')})
        variants = {}
        for variant in ('original', 'distilled'):
            path = directory / f'{variant}.json'
            if not path.exists():
                continue
            value = read_json(path)
            identity = (value['test_sha256'], value['count'], value['settings'], value['backend'], value.get('versions'))
            if reference is None:
                reference = identity
            if identity != reference:
                raise ValueError(f'Cannot mix different test sets, runtimes or decoding settings: {path}')
            if value['model_id'] != spec['id'] or value['variant'] != variant:
                raise ValueError(f'Wrong model or variant in report: {path}')
            value['label'] = spec['label']
            variants[variant] = value
            results.append(value)
        if len(variants) == 2:
            old, new = variants['original'], variants['distilled']
            if old['revision'] != new['revision'] or old['chat_template_sha256'] != new['chat_template_sha256']:
                raise ValueError(f'Original/distilled revision or template mismatch: {spec["id"]}')
            pairs.append({'id': spec['id'], 'label': spec['label'],
                          'delta_distilled_minus_original': {k: new[k] - old[k] for k in METRICS},
                          'score_mae_delta': {k: new['score_mae'][k] - old['score_mae'][k]
                                              for k in ('etki', 'ciddiyet', 'aciliyet')}})
    results.sort(key=lambda x: (-x['exact_match'], -x['topic_micro_f1'], x['model_id'], x['variant']))
    report = {'schema_version': 1, 'results': results, 'pairs': pairs, 'statuses': statuses,
              'expected_variants': len(config['models']) * 2, 'completed_variants': len(results),
              'notes': ['Teacher-label agreement on a small, previously evaluated test set; not human accuracy.',
                        'All results use HF/CUDA bfloat16 greedy generation without a JSON grammar.',
                        'Timing is amortized batch inference time per headline, not serial latency or phone speed.',
                        'Native wrappers differ across models. Within each pair, wrapper and checkpoint revision match.',
                        'Failed or pending variants are not assigned zero scores.']}
    score_report(report)
    write_json(output / 'leaderboard.json', report)
    fields = ['model', 'variant', 'overall_score', 'overall_score_version', 'exact_match', 'topic_micro_f1', 'type_accuracy', 'impact_mae',
              'severity_mae', 'urgency_mae', 'invalid', 'amortized_ms_per_headline', 'parameters', 'test_count']
    with (output / 'leaderboard.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for value in results:
            writer.writerow(dict(model=value['model'], variant=value['variant'],
                                  overall_score=value['overall_score'], overall_score_version=value['overall_score_version'],
                                  **{k: value[k] for k in METRICS},
                                  impact_mae=value['score_mae']['etki'], severity_mae=value['score_mae']['ciddiyet'],
                                  urgency_mae=value['score_mae']['aciliyet'],
                                  amortized_ms_per_headline=value['latency_seconds']['mean'] * 1000,
                                  parameters=value['parameters'], test_count=value['count']))
    lines = ['# Multi-model distillation comparison', '',
             f'Completed: **{len(results)} / {len(config["models"]) * 2}** variants.', '',
             'Gündem Score v1 (0–100): 30% topic F1 + 20% type accuracy + '
             '30% normalized score accuracy + 20% exact match. Score accuracy = 1 − mean MAE / 3. '
             'Quality only; hardware speed is separate. Invalid outputs are already penalized. '
             'This is a task-specific composite, not an accuracy percentage.', '',
             '| Model | Variant | Gündem Score / 100 | Exact match | Topic F1 | Type accuracy | Invalid |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for v in results:
        lines.append(f'| {v["label"]} | {v["variant"]} | {v["overall_score"]:.1f} | {v["exact_match"]:.1%} | '
                     f'{v["topic_micro_f1"]:.1%} | {v["type_accuracy"]:.1%} | {v["invalid"]} |')
    lines += ['', '## Model status', '', '| Model | Status | Detail |', '| --- | --- | --- |']
    for state in statuses:
        detail = state['detail'].replace('|', '/').replace('\n', ' ')[:300]
        lines.append(f'| {state["label"]} | {state["status"]} | {detail} |')
    lines += ['', '## Interpretation', '', *['- ' + note for note in report['notes']]]
    (output / 'leaderboard.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    if not report['results']:
        return []
    score_report(report)
    ids = [s['id'] for s in report['statuses']]
    names = [s['label'] for s in report['statuses']]
    lookup = {(v['model_id'], v['variant']): v for v in report['results']}
    colors = ['#64748b', '#0d9488']
    outputs = []
    for lang in ('tr', 'en'):
        tr = lang == 'tr'
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
        fig, axes = plt.subplots(2, 3, figsize=(20, max(10, len(ids) * .65 + 4)))
        fig.patch.set_facecolor('#f8fafc')
        fig.subplots_adjust(left=.15, right=.98, top=.84, bottom=.18, wspace=.53, hspace=.22)
        fig.text(.15, .955, 'Çoklu model · Damıtma kıyaslaması' if tr else 'Multi-model · Distillation comparison',
                 fontsize=24, weight='bold', color='#0f172a')
        sample = report['results'][0]
        fig.text(.15, .922, f'{report["completed_variants"]}/{report["expected_variants"]} '
                 + ('tamamlanan varyant' if tr else 'completed variants')
                 + f' · n={sample["count"]} · HF/CUDA BF16 · batch={sample["settings"]["batch_size"]} · greedy',
                 fontsize=12, color='#475569')
        fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(colors,
                   ['Orijinal', 'Distilled'] if tr else ['Original', 'Distilled'])],
                   loc='upper left', bbox_to_anchor=(.146, .9), ncol=2, frameon=False)
        specs = [
            ('Tam eşleşme ↑' if tr else 'Exact match ↑', lambda r: r['exact_match'] * 100, 100, '%'),
            ('Etiket micro-F1 ↑' if tr else 'Topic micro-F1 ↑', lambda r: r['topic_micro_f1'] * 100, 100, '%'),
            ('Tür doğruluğu ↑' if tr else 'Type accuracy ↑', lambda r: r['type_accuracy'] * 100, 100, '%'),
            ('Ortalama puan MAE ↓' if tr else 'Mean score MAE ↓', lambda r: sum(r['score_mae'].values()) / 3, 3, 'MAE'),
            ('Geçersiz cevap ↓' if tr else 'Invalid responses ↓', lambda r: r['invalid'], sample['count'], 'Adet' if tr else 'Count'),
            ('Amorti süre / başlık ↓' if tr else 'Amortized time / headline ↓', lambda r: r['latency_seconds']['mean'] * 1000,
             None, 'ms (batch; seri gecikme değil)' if tr else 'ms (batch; not serial latency)')]
        for index, (ax, (title, getter, maximum, unit)) in enumerate(zip(axes.flat, specs)):
            y = np.arange(len(ids))
            ax.set_facecolor('#f8fafc')
            ax.set_title(title, loc='left', weight='bold', fontsize=12)
            ax.set_axisbelow(True)
            ax.grid(axis='x', color='#e2e8f0')
            ax.set_xlabel(unit, fontsize=9)
            for j, variant in enumerate(('original', 'distilled')):
                for i, model_id in enumerate(ids):
                    value = lookup.get((model_id, variant))
                    position = i + (j - .5) * .34
                    if value:
                        metric = getter(value)
                        ax.barh(position, metric, height=.29, color=colors[j])
                        formatted = f'{metric:.0f}' if index == 4 else f'{metric:.2f}' if index == 3 else f'{metric:.1f}'
                        ax.annotate(formatted, (metric, position), xytext=(3, 0),
                                    textcoords='offset points', va='center', fontsize=8, color=colors[j])
                    else:
                        ax.text(.01, position, '—', transform=ax.get_yaxis_transform(), va='center', color='#94a3b8')
            ax.set_yticks(y, names if index in (0, 3) else [])
            ax.set_ylim(len(ids) - .5, -.5)
            values = [getter(v) for v in report['results']]
            ax.set_xlim(0, maximum * 1.15 if maximum else max(max(values) * 1.25, 1))
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis='y', length=0)
        fig.text(.15, .055, ('Öğretmen etiketleriyle uyum; telefon performansı değildir. — = sonuç yok, sıfır puan değil.\n'
                            'Bütün varyantlar aynı testte; şablonlar model bazında kaydedilir. Başarısızlık ayrıntıları: leaderboard.md.'
                            if tr else 'Agreement with teacher labels; not phone performance. — = no result, not a zero score.\n'
                            'All variants share the test set; wrappers are recorded per model. Failure details: leaderboard.md.'),
                 fontsize=10, color='#64748b')
        for ext in ('png', 'svg'):
            path = Path(output) / f'comparison-{lang}.{ext}'
            fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
            outputs.append(path)
        plt.close(fig)
        # A shareable single-number view, while keeping the full six-panel chart.
        ordered = sorted(report['statuses'], key=lambda s: -max(
            (r['overall_score'] for r in report['results'] if r['model_id'] == s['id']), default=-1))
        fig, ax = plt.subplots(figsize=(12, max(6, len(ordered) * .55 + 2.5)))
        fig.patch.set_facecolor('#f8fafc')
        ax.set_facecolor('#f8fafc')
        fig.subplots_adjust(left=.27, right=.96, top=.79, bottom=.22)
        fig.text(.27, .94, 'Gündem Score · v1', fontsize=24, weight='bold', color='#0f172a')
        fig.text(.27, .89, ('Haber sınıflandırma kalitesi · 0–100 · yüksek daha iyi' if tr else
                           'Headline classification quality · 0–100 · higher is better'), color='#475569')
        for j, variant in enumerate(('original', 'distilled')):
            for i, spec in enumerate(ordered):
                result = lookup.get((spec['id'], variant))
                y = i + (j - .5) * .34
                if result:
                    value = result['overall_score']
                    ax.barh(y, value, height=.29, color=colors[j])
                    ax.annotate(f'{value:.1f}', (value, y), xytext=(4, 0),
                                textcoords='offset points', va='center', fontsize=10, color=colors[j])
                else:
                    ax.text(.01, y, '—', transform=ax.get_yaxis_transform(), color='#94a3b8')
        ax.set_yticks(range(len(ordered)), [s['label'] for s in ordered])
        ax.set_ylim(len(ordered) - .5, -.5)
        ax.set_xlim(0, 110)
        ax.set_xticks(range(0, 101, 20))
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#e2e8f0')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(colors,
                  ['Orijinal', 'Distilled'] if tr else ['Original', 'Distilled'])],
                  bbox_to_anchor=(0, 1.14), loc='upper left', frameon=False, ncol=2)
        formula = ('30 F1 + 20 tür + 30 (1 − ort. MAE / 3) + 20 tam eşleşme' if tr else
                   '30 F1 + 20 type + 30 (1 − mean MAE / 3) + 20 exact match')
        fig.text(.27, .13, formula, fontsize=10, color='#475569')
        fig.text(.27, .065, (f'n={sample["count"]} · Öğretmen etiketleriyle uyum; genel LLM puanı veya doğruluk yüzdesi değildir.\n'
                            'Hız dahil değil. — = ölçüm yok. Ayrıntılar: leaderboard.md.' if tr else
                            f'n={sample["count"]} · Teacher-label agreement; not general LLM ability or accuracy percent.\n'
                            'Speed excluded. — = no evaluation. Details: leaderboard.md.'), fontsize=9, color='#64748b')
        for ext in ('png', 'svg'):
            path = Path(output) / f'overall-{lang}.{ext}'
            fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
            outputs.append(path)
        plt.close(fig)
    return outputs
