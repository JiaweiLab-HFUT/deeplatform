from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from core.register.registryhub import RegistryHub
from frame.frame import Frame


class FrameCheckTests(unittest.TestCase):
    def setUp(self):
        self.frame = Frame()
        self.frame.register_hub = RegistryHub()
        self.frame.register_hub.task.register('TestTask')(lambda context, hook_manager: context)
        for scope, names in {'model': ['TestModel'], 'metric': ['Metric', 'OtherMetric'],
                             'validate': ['Validator'], 'later': ['Later']}.items():
            for name in names:
                self.frame.register_hub[scope].register(name)(lambda **params: params)
        self.config = {
            'mate': {'seed': 42, 'experiment': 'test'},
            'task': {'type': 'TestTask'},
            'model': {'type': 'TestModel'},
        }

    def test_valid_config_and_nested_references_are_not_modified(self):
        self.config['metric'] = [
            {'type': 'Metric'},
            {'type': 'Metric', 'name': 'other_metric'},
        ]
        self.config['validate'] = {
            'type': 'Validator',
            'params': {'nested': ['@model', {'metrics': ('@Metric', '@other_metric')}],
                       'seed': '@seed'},
        }
        original = deepcopy(self.config)
        self.frame.check(self.config)
        self.assertEqual(self.config, original)

    def test_invalid_structure_reports_config_location(self):
        cases = [
            (None, '顶层'),
            ({}, 'mate'),
            ({'mate': []}, 'mate'),
        ]
        for config, message in cases:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, message):
                    self.frame.check(config)
        mutations = [
            ('task', None, 'task'),
            ('task', {}, 'task.type'),
            ('model', 1, 'model'),
            ('model', [None], r'model\[0\]'),
            ('model', {'type': ''}, 'model.type'),
            ('model', {'type': 'Model', 'name': 3}, 'model.name'),
            ('model', {'type': 'Model', 'params': None}, 'model.params'),
            ('model', {'type': 'Model', 'params': {1: 'value'}}, '参数名'),
        ]
        for field, value, message in mutations:
            with self.subTest(field=field, value=value):
                config = deepcopy(self.config)
                config[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.frame.check(config)

    def test_seed_is_required_and_accepts_only_numpy_compatible_integers(self):
        for seed in [None, True, '42', 1.5, -1, 2**32]:
            with self.subTest(seed=seed):
                self.config['mate']['seed'] = seed
                with self.assertRaisesRegex(ValueError, 'mate.seed'):
                    self.frame.check(self.config)
        for seed in [0, 2**32 - 1]:
            self.config['mate']['seed'] = seed
            self.frame.check(self.config)

    def test_metadata_cannot_shadow_scopes_or_framework_names(self):
        for name in ['MODEL', 'task', '__experiment_path__']:
            with self.subTest(name=name):
                config = deepcopy(self.config)
                config['mate'][name] = 'value'
                with self.assertRaisesRegex(ValueError, '冲突'):
                    self.frame.check(config)
        self.frame.register_hub['optimizer']
        self.config['mate']['optimizer'] = 'value'
        with self.assertRaisesRegex(ValueError, '冲突'):
            self.frame.check(self.config)

    def test_duplicate_scope_names_are_case_insensitive(self):
        self.config['MODEL'] = {'type': 'Model'}
        with self.assertRaisesRegex(ValueError, '注册域'):
            self.frame.check(self.config)

    def test_duplicate_names_include_defaults_and_cross_scope_collisions(self):
        duplicate_configs = [
            {'metric': [{'type': 'Metric'}, {'type': 'Metric'}]},
            {'metric': [{'type': 'Metric', 'name': 'shared'},
                        {'type': 'OtherMetric', 'name': 'shared'}]},
            {'metric': {'type': 'Metric', 'name': 'model'}},
            {'metric': {'type': 'Metric', 'name': 'seed'}},
            {'metric': {'type': 'Metric', 'name': 'task'}},
        ]
        for extra in duplicate_configs:
            with self.subTest(extra=extra):
                config = deepcopy(self.config)
                config.update(extra)
                with self.assertRaisesRegex(ValueError, '重复'):
                    self.frame.check(config)

    def test_unknown_and_self_references_fail_before_building(self):
        for reference, message in [('@missing', '未定义'),
                                   ('@model', '不能引用自身')]:
            with self.subTest(reference=reference):
                config = deepcopy(self.config)
                config['model']['params'] = {'nested': [{'dependency': reference}]}
                config['later'] = {'type': 'Later'}
                with self.assertRaisesRegex(ValueError, message) as error:
                    self.frame.check(config)
                if reference == '@missing':
                    self.assertIn('model.params.nested[0].dependency', str(error.exception))
                    self.assertIn('引用 @missing', str(error.exception))

    def test_forward_references_are_ordered_before_their_consumers(self):
        self.config['model']['params'] = {'nested': [{'dependency': '@later'}]}
        self.config['later'] = {'type': 'Later'}
        plan = self.frame.check(self.config)
        self.assertEqual([item.spec.name for item in plan.components], ['later', 'model'])


class FrameSeedTests(unittest.TestCase):
    def test_seed_controls_model_initialization_and_training_randomness(self):
        hub = RegistryHub()
        results = []

        @hub.model.register
        class RandomModel:
            def __init__(self):
                self.python_value = random.random()
                self.numpy_value = float(np.random.random())
                self.layer = torch.nn.Linear(3, 2)

        @hub.task.register
        class RecordingTask:
            def __init__(self, context, hook_manager):
                self.context = context

            def run(self):
                model = self.context.model
                results.append((
                    model.python_value, model.numpy_value,
                    model.layer.weight.detach().tolist(),
                    model.layer.bias.detach().tolist(),
                    random.random(), float(np.random.random()), torch.rand(3).tolist(),
                ))

        def discover():
            # 即使模块导入消耗随机数，也不影响随后的模型初始化。
            random.random()
            np.random.random()
            torch.rand(1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            for seed in [42, 42, 43]:
                path.write_text(yaml.safe_dump({
                    'mate': {'seed': seed, 'experiment': 'test',
                             'experiment_root': str(Path(directory) / 'runs')},
                    'task': {'type': 'RecordingTask'},
                    'model': {'type': 'RandomModel'},  # params 可以省略
                }), encoding='utf-8')
                frame = Frame()
                frame.register_hub = hub
                frame.components_builder.register_hub = hub
                with patch('frame.frame.autodiscover_components', side_effect=discover):
                    frame.run(path)

        self.assertEqual(results[0], results[1])
        self.assertNotEqual(results[0], results[2])


if __name__ == '__main__':
    unittest.main()
