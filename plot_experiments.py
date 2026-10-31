#!/usr/bin/env python3
"""Render bilingual multi-model figures from a saved leaderboard JSON only."""
import argparse
from pathlib import Path
from lab.common import ROOT, read_json
from lab.report import plot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'workspace/reports/leaderboard.json')
    p.add_argument('--out', type=Path, help='Output directory (default: alongside input)')
    a = p.parse_args()
    try:
        output = a.out or a.input.parent
        output.mkdir(parents=True, exist_ok=True)
        paths = plot(read_json(a.input), output)
        for path in paths:
            print(path.resolve())
        if not paths:
            print('No completed results to plot.')
    except ImportError:
        p.exit(1, 'Install Matplotlib in this environment: python3 -m pip install -r requirements-report.txt\n')
    except (OSError, ValueError, KeyError, TypeError) as error:
        p.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
