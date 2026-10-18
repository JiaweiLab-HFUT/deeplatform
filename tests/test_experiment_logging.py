from contextlib import redirect_stderr, redirect_stdout
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
import unittest

from core.experiment.logging import experiment_logging
from frame.frame import Frame


class ExperimentLoggingTests(unittest.TestCase):
    def test_captures_streams_logging_and_success_status_and_restores_state(self):
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        with tempfile.TemporaryDirectory() as directory:
            terminal = io.StringIO()
            with redirect_stdout(terminal), redirect_stderr(terminal):
                original_stdout, original_stderr = sys.stdout, sys.stderr
                with experiment_logging(directory, {'seed': 42}):
                    print('training output')
                    print('progress output', file=sys.stderr)
                    logging.warning('warning output')
                    status = json.loads((Path(directory) / 'status.json').read_text())
                    self.assertEqual(status['status'], 'running')
                self.assertIs(sys.stdout, original_stdout)
                self.assertIs(sys.stderr, original_stderr)
            text = (Path(directory) / 'train.log').read_text()
            for message in ['training output', 'progress output', 'warning output']:
                self.assertEqual(text.count(message), 1)
                self.assertIn(message, terminal.getvalue())
            status = json.loads((Path(directory) / 'status.json').read_text())
            self.assertEqual(status['status'], 'completed')
            self.assertEqual(status['seed'], 42)
            self.assertIn('finished_at', status)
        self.assertEqual(root.handlers, handlers)
        self.assertEqual(root.level, level)

    def test_failure_is_recorded_with_traceback_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'training failed'):
                    with experiment_logging(directory):
                        raise RuntimeError('training failed')
            text = (Path(directory) / 'train.log').read_text()
            self.assertIn('Traceback', text)
            self.assertIn('RuntimeError: training failed', text)
            status = json.loads((Path(directory) / 'status.json').read_text())
            self.assertEqual(status['status'], 'failed')

    def test_experiments_have_separate_directories_and_config_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                'mate': {'seed': 42, 'experiment': 'test', 'experiment_root': directory},
                'task': {'type': 'TestTask'},
            }
            frame = Frame()
            frame.build_mate_task(config)
            frame.creat_experiment('unused.yaml', config)
            first = Path(frame.context.__experiment_path__)
            frame.creat_experiment('unused.yaml', config)
            second = Path(frame.context.__experiment_path__)
            self.assertNotEqual(first, second)
            for path in [first, second]:
                self.assertEqual(path.parent, Path(directory))
                self.assertIn('experiment_root', (path / 'config.yaml').read_text())


if __name__ == '__main__':
    unittest.main()
