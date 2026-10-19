import json
from pathlib import Path
import signal
import os
import tempfile
import unittest

import yaml

from main import launch_background


class BackgroundTests(unittest.TestCase):
    def test_background_run_keeps_all_artifacts_in_one_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'experiment'
            source = Path(temporary) / 'source.yaml'
            source.write_text(yaml.safe_dump({
                'mate': {'seed': 42, 'experiment': 'background', 'experiment_root': str(root)},
                'task': {'type': 'MyTask'},
                'hook': [{'type': 'PrintStart', 'params': {'priority': 1}},
                         {'type': 'PrintEnd'}],
            }), encoding='utf-8')
            process, directory = launch_background(source)
            try:
                code = process.wait(timeout=45)
                console = (directory / 'console.log').read_text()
                self.assertEqual(code, 0, console)
                self.assertEqual(list(root.iterdir()), [directory])
                for name in ['config.yaml', 'console.log', 'train.log', 'status.json', 'checkpoint']:
                    self.assertTrue((directory / name).exists(), name)
                state = json.loads((directory / 'status.json').read_text())
                self.assertEqual(state['pid'], process.pid)
                self.assertEqual(state['status'], 'completed')
                self.assertIn('开始', console)
                self.assertIn('结束', (directory / 'train.log').read_text())
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
