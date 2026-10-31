#!/usr/bin/env python3
"""Render old/new benchmark metrics as PNG and SVG, without rerunning models."""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = {
    'tr': {
        'title': 'LFM · Orijinal ve Distilled',
        'subtitle': '{count} ayrılmış test başlığı · Aynı test kümesi ve öğrenci prompt’u',
        'models': ['Orijinal', 'Distilled'],
        'quality': 'Kalite · Yüksek daha iyi',
        'quality_names': ['Tam eşleşme', 'Etiket micro-F1', 'Tür doğruluğu'],
        'error': 'Puan hatası · Düşük daha iyi',
        'score_names': ['Etki', 'Ciddiyet', 'Aciliyet'],
        'mae': 'Ortalama mutlak hata (MAE)',
        'latency': 'İstek süresi · Düşük daha iyi',
        'latency_names': ['Ortalama', 'p50', 'p95'], 'ms': 'Milisaniye',
        'invalid': 'Geçersiz cevap · Düşük daha iyi',
        'invalid_names': ['Geçersiz / tamamlanmamış JSON'], 'count': 'Adet / {count}',
        'note': 'Kalite: referans etiketlerle uyum. Süre: bu çalışma ortamındaki istek süresi.',
        'source': 'Kaynak',
    },
    'en': {
        'title': 'LFM · Original vs. Distilled',
        'subtitle': '{count} held-out test headlines · Same test set and student prompt',
        'models': ['Original', 'Distilled'],
        'quality': 'Quality · Higher is better',
        'quality_names': ['Exact match', 'Topic micro-F1', 'Type accuracy'],
        'error': 'Score error · Lower is better',
        'score_names': ['Impact', 'Severity', 'Urgency'],
        'mae': 'Mean absolute error (MAE)',
        'latency': 'Request latency · Lower is better',
        'latency_names': ['Mean', 'p50', 'p95'], 'ms': 'Milliseconds',
        'invalid': 'Invalid responses · Lower is better',
        'invalid_names': ['Invalid / incomplete JSON'], 'count': 'Count / {count}',
        'note': 'Quality: agreement with reference labels. Latency: request time in this environment.',
        'source': 'Source',
    },
}


def render(source, output=None, lang='tr', comparison=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    labels = LABELS[lang].copy()
    source = Path(source)
    report = json.loads(source.read_text(encoding='utf-8'))
    comparison = comparison or report.get('comparison', 'original-distilled')
    if comparison == 'starting-checkpoint':
        labels['title'] = ('LFM3 · Başlangıç modeli kıyaslaması' if lang == 'tr'
                           else 'LFM3 · Starting checkpoint comparison')
        labels['models'] = ['Post-trained + LoRA', 'Base + LoRA']
    models = [report['old'], report['new']]
    count = report['count']
    if not isinstance(count, int) or count <= 0 or any(m['count'] != count for m in models):
        raise ValueError('Model test counts must match and be positive')
    output = Path(output) if output else source.with_name(source.stem + '-comparison')
    if output.suffix in ('.png', '.svg'):
        output = output.with_suffix('')
    if lang == 'en':
        output = output.with_name(output.name + '-en')
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = ['#64748b', '#0d9488']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.edgecolor': '#cbd5e1',
                         'text.color': '#0f172a', 'axes.labelcolor': '#475569',
                         'xtick.color': '#475569', 'ytick.color': '#475569',
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#f8fafc')
    fig.subplots_adjust(left=.07, right=.97, top=.79, bottom=.19, hspace=.55, wspace=.22)
    fig.text(.07, .94, labels['title'], fontsize=25, weight='bold')
    fig.text(.07, .898, labels['subtitle'].format(count=count),
             fontsize=12, color='#475569')
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(colors, labels['models'])],
               loc='upper left', bbox_to_anchor=(.063, .88), frameon=False, ncol=2)

    def panel(ax, title, keys, names, getter, limit, unit, fmt):
        ax.set_facecolor('#f8fafc')
        ax.set_title(title, loc='left', fontsize=13, weight='bold', pad=18)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e2e8f0')
        ax.tick_params(axis='both', length=0, pad=8)
        ax.set_ylabel(unit)
        positions = list(range(len(keys)))
        for idx, model in enumerate(models):
            values = [getter(model, key) for key in keys]
            if any(not math.isfinite(v) or v < 0 or (limit is not None and v > limit) for v in values):
                raise ValueError(f'Invalid metric values: {title}')
            bars = ax.bar([p + (idx - .5) * .34 for p in positions], values,
                          width=.30, color=colors[idx], zorder=3)
            ax.bar_label(bars, labels=[fmt(v) for v in values], padding=5, fontsize=10,
                         color=colors[idx], weight='bold')
        ax.set_xticks(positions, names)
        maximum = max(getter(m, k) for m in models for k in keys)
        ax.set_ylim(0, limit if limit is not None else max(maximum * 1.3, 1))
        ax.set_xlim(-.6, len(keys) - .4)

    panel(axes[0, 0], labels['quality'],
          ['exact_match', 'topic_micro_f1', 'type_accuracy'],
          labels['quality_names'],
          lambda m, k: m[k] * 100, 100, '%',
          lambda v: f'{v:.1f}%' if lang == 'en' else f'%{v:.1f}')
    panel(axes[0, 1], labels['error'],
          ['etki', 'ciddiyet', 'aciliyet'], labels['score_names'],
          lambda m, k: m['score_mae'][k], 3, labels['mae'], lambda v: f'{v:.2f}')
    panel(axes[1, 0], labels['latency'],
          ['mean', 'p50', 'p95'], labels['latency_names'],
          lambda m, k: m['latency_seconds'][k] * 1000, None, labels['ms'], lambda v: f'{v:.0f}')
    panel(axes[1, 1], labels['invalid'],
          ['invalid'], labels['invalid_names'],
          lambda m, k: m[k], None, labels['count'].format(count=count), lambda v: f'{v:.0f}')
    axes[1, 1].yaxis.get_major_locator().set_params(integer=True)
    details = [f'{name}: {model["model"]}' for name, model in zip(labels['models'], models)]
    if report.get('backend'):
        details.append(report['backend'])
    details.extend([labels['note'], f'{labels["source"]}: {source.name}'])
    fig.text(.07, .135, '\n'.join(details), fontsize=9, color='#64748b', linespacing=1.5, va='top')
    paths = [Path(str(output) + suffix) for suffix in ('.png', '.svg')]
    for path in paths:
        fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / 'data-deepseek/benchmark-lms.json')
    parser.add_argument('--out', type=Path, help='Output filename prefix (PNG and SVG are both generated)')
    parser.add_argument('--lang', choices=['both', 'tr', 'en'], default='both',
                        help='Chart language (default: both; English files have an -en suffix)')
    parser.add_argument('--comparison', choices=['original-distilled', 'starting-checkpoint'],
                        help='Comparison labels (default: read report, otherwise original/distilled)')
    args = parser.parse_args()
    try:
        for lang in (['tr', 'en'] if args.lang == 'both' else [args.lang]):
            for path in render(args.input, args.out, lang=lang, comparison=args.comparison):
                print(f'Chart ({lang}): {path.resolve()}')
    except ImportError:
        parser.exit(1, 'Matplotlib gerekli. Kurulum: python3 -m pip install matplotlib\n')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
