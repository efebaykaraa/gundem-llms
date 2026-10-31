"""Terminal save-and-exit control shared by the launcher and labeling command."""
from contextlib import contextmanager
import os
from pathlib import Path
import select
import sys
import tempfile
import termios
import threading
import tty


class SavedExit(Exception):
    """Work was saved and the user requested an orderly exit."""


_active = None


class StopControl:
    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdin
        self.requested = threading.Event()
        self.closed = threading.Event()
        self.thread = None
        self.fd = None
        self.settings = None

    def request(self):
        # The training child observes the same request at an optimizer step boundary.
        self.stop_file.touch()
        self.requested.set()
        print('\nQ: Kaydedip çıkma istendi; devam eden işler kaydediliyor…', file=sys.stderr, flush=True)

    def check(self):
        if self.requested.is_set():
            raise SavedExit()

    def _listen(self):
        while not self.closed.is_set():
            ready, _, _ = select.select([self.fd], [], [], 0.1)
            if ready:
                key = os.read(self.fd, 1)
                if not key:
                    return
                if key in (b'q', b'Q'):
                    self.request()
                    return

    def __enter__(self):
        self.directory = tempfile.TemporaryDirectory(prefix='gundem-stop-')
        self.stop_file = Path(self.directory.name) / 'stop'
        try:
            if self.stream.isatty():
                self.fd = self.stream.fileno()
                self.settings = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
                self.thread = threading.Thread(target=self._listen, daemon=True)
                self.thread.start()
                print('Kaydedip çık: Q (Enter gerekmez).', file=sys.stderr, flush=True)
        except (AttributeError, OSError, termios.error):
            if self.settings is not None:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)
            self.settings = None
        return self

    def __exit__(self, *exc):
        self.closed.set()
        if self.thread:
            self.thread.join()
        if self.settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)
        self.directory.cleanup()


@contextmanager
def stop_session():
    global _active
    if _active is not None:
        yield _active
        return
    with StopControl() as control:
        _active = control
        try:
            yield control
        finally:
            _active = None


def current_control():
    return _active
