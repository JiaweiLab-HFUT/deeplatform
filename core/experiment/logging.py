"""将一次实验的终端输出、日志和运行状态保存在实验目录。"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys


class _Tee:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, text):
        self.logfile.write(text)
        self.logfile.flush()
        return self.terminal.write(text)

    def flush(self):
        self.logfile.flush()
        self.terminal.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@contextmanager
def experiment_logging(directory, metadata=None):
    """同时输出到终端和 train.log，退出时恢复标准流及 logging 配置。"""
    directory = Path(directory)
    status_path = directory / 'status.json'
    status = dict(metadata or {})
    status.update(pid=os.getpid(), status='running',
                  started_at=datetime.now(timezone.utc).isoformat())

    def save_status():
        temporary = status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(status_path)

    save_status()
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    with (directory / 'train.log').open('a', encoding='utf-8', buffering=1) as logfile:
        with redirect_stdout(_Tee(sys.stdout, logfile)), redirect_stderr(_Tee(sys.stderr, logfile)):
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
            root.handlers = [handler]
            root.setLevel(logging.INFO)
            try:
                logging.info('Experiment started: %s (PID %s)', directory, os.getpid())
                yield
            except BaseException as exc:
                status['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
                status['error'] = f'{type(exc).__name__}: {exc}'
                logging.exception('Experiment %s', status['status'])
                raise
            else:
                status['status'] = 'completed'
                logging.info('Experiment completed')
            finally:
                status['finished_at'] = datetime.now(timezone.utc).isoformat()
                root.handlers = old_handlers
                root.setLevel(old_level)
                handler.close()
                save_status()
