"""Responsive terminal progress. Only completed valid labels advance the percentage."""
import sys
import shutil
import threading
import time


def progress_mode(mode, stream=None):
    stream = stream or sys.stderr
    return ('bar' if stream.isatty() else 'plain') if mode == 'auto' else mode


class StageProgress:
    """Show liveness and elapsed time when the loader provides no measurable total."""
    def __init__(self, label, mode='auto', stream=None):
        self.label = label
        self.stream = stream or sys.stderr
        self.mode = progress_mode(mode, self.stream)
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.thread = None
        self.frame = 0

    def render(self, status=None):
        if self.mode == 'off':
            return
        elapsed = int(time.monotonic() - self.started)
        marker = status or '|/-\\'[self.frame % 4]
        self.frame += 1
        line = f'{self.label} [{marker}] {elapsed//60:02d}:{elapsed%60:02d}'
        prefix = '\r\033[2K' if self.mode == 'bar' else ''
        self.stream.write(prefix + line + ('\n' if self.mode != 'bar' or status else ''))
        self.stream.flush()

    def _heartbeat(self):
        while not self.stop.wait(0.5 if self.mode == 'bar' else 10):
            self.render()

    def __enter__(self):
        self.render('başladı' if self.mode != 'bar' else None)
        if self.mode != 'off':
            self.thread = threading.Thread(target=self._heartbeat, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, kind, value, tb):
        self.stop.set()
        if self.thread:
            self.thread.join()
        self.render('durduruldu' if kind else 'tamamlandı')


def download_tqdm(mode):
    """Explicit tqdm class keeps HF byte progress visible through the launcher pipe."""
    from tqdm.std import tqdm
    mode = progress_mode(mode)

    class DownloadProgress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.update(disable=mode == 'off', file=sys.stderr,
                          dynamic_ncols=mode == 'bar', mininterval=0.2 if mode == 'bar' else 5)
            super().__init__(*args, **kwargs)

        def display(self, msg=None, pos=None):
            if mode == 'plain':
                self.fp.write((str(self) if msg is None else msg) + '\n')
                self.fp.flush()
            else:
                super().display(msg, pos)

    return DownloadProgress


class TeacherProgress:
    def __init__(self, total, reused=0, mode='auto', stream=None):
        self.stream = stream or sys.stderr
        self.mode = ('bar' if self.stream.isatty() else 'plain') if mode == 'auto' else mode
        self.total = total
        self.valid = reused
        self.errors = 0
        self.active = 0
        self.status = 'hazır'
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.last_plain = None

    def line(self):
        fraction = self.valid / self.total if self.total else 1.0
        elapsed = int(time.monotonic() - self.started)
        width = shutil.get_terminal_size(fallback=(120, 20)).columns
        status = self.status
        if width < 100 and status.startswith('işleniyor '):
            status = 'işleniyor'
        tail = (f' %{100*fraction:5.1f} {self.valid}/{self.total}'
                f' | kalan {self.total-self.valid} | aktif {self.active} | hata {self.errors}'
                f' | {elapsed//60:02d}:{elapsed%60:02d} | {status}')
        size = max(3, min(24, width - len('Öğretmen []') - len(tail) - 1))
        filled = int(size * fraction)
        bar = '█' * filled + '░' * (size - filled)
        return (f'Öğretmen [{bar}]' + tail)[:max(1, width-1)]

    def render(self):
        if self.mode == 'off':
            return
        with self.lock:
            if self.mode == 'bar':
                self.stream.write('\r\033[2K' + self.line())
            else:
                key = (self.valid, self.errors, self.active, self.status)
                if key == self.last_plain:
                    return
                self.last_plain = key
                self.stream.write(self.line() + '\n')
            self.stream.flush()

    def _heartbeat(self):
        while not self.stop.wait(0.5):
            self.render()

    def begin(self, identifier):
        with self.lock:
            self.active += 1
            self.status = 'işleniyor ' + identifier[:8]
            self.render()

    def retry(self, attempt):
        with self.lock:
            self.status = f'yeniden deneme {attempt}'
            self.render()

    def success(self):
        with self.lock:
            self.valid += 1
            self.active = max(0, self.active - 1)
            self.status = 'kaydedildi'
            self.render()

    def failure(self):
        with self.lock:
            self.errors += 1
            self.active = max(0, self.active - 1)
            self.status = 'hata; loga kaydedildi'
            self.render()

    def __enter__(self):
        self.render()
        if self.mode == 'bar':
            self.thread = threading.Thread(target=self._heartbeat, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, kind, value, tb):
        self.stop.set()
        if self.thread:
            self.thread.join()
        with self.lock:
            if kind:
                self.status = 'durduruldu'
            elif self.valid == self.total:
                self.status = 'tamamlandı'
            else:
                self.status = 'bu tur bitti'
            self.active = 0
            self.render()
            if self.mode == 'bar':
                self.stream.write('\n')
                self.stream.flush()
