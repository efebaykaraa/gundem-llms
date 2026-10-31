"""Download pinned checkpoints ahead of the sequential GPU queue."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import time
from .common import ROOT, read_json, write_json


def main():
    from train import configure_hub_download, download_student
    configure_hub_download()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=ROOT / 'workspace')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    if not 1 <= a.workers <= 4:
        p.error('workers must be between 1 and 4')
    specs = [read_json(path) for path in sorted(a.workspace.glob('experiments/*/model.lock.json'))]
    def fetch(spec):
        print('DOWNLOAD', spec['id'], flush=True)
        for attempt in range(3):
            try:
                path = download_student(spec['repo'], spec['revision'], 'plain')
                return spec['id'], {'status': 'downloaded', 'revision': spec['revision'], 'path': path}
            except Exception as error:
                if attempt == 2:
                    return spec['id'], {'status': 'failed', 'error': type(error).__name__}
                time.sleep(2)
    completed = {}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(fetch, spec) for spec in specs]):
            key, result = future.result()
            completed[key] = result
            write_json(a.workspace / 'downloads.json', completed)
            print(key, result['status'], flush=True)


if __name__ == '__main__':
    main()
