from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.builder.builder import ComponentsBuilder
from core.config.schema import parse_config
from core.context.context import Context
from core.register.registryhub import RegistryHub
from core.register.register import Registry
from frame.frame import Frame


class ComponentConfigTests(unittest.TestCase):
    def setUp(self):
        self.hub = RegistryHub()
        self.context = Context().bind('test')
        self.builder = ComponentsBuilder(self.context, self.hub)
        self.frame = Frame()
        self.frame.register_hub = self.hub
        self.hub.task.register('Task')(lambda context, hook_manager: context)
        self.cfg = {'mate': {'seed': 42, 'experiment': 'test'},
                    'task': {'type': 'Task'}}

    def test_unknown_description_fields_are_rejected(self):
        for field in ['param', 'parmas', 'depend_on', 'extra']:
            with self.subTest(field=field):
                cfg = deepcopy(self.cfg)
                cfg['model'] = {'type': 'Model', field: {}}
                with self.assertRaisesRegex(ValueError, 'model.*未知字段'):
                    self.frame.check(cfg)
        self.cfg['task']['name'] = 'ignored_before'
        with self.assertRaisesRegex(ValueError, 'task.*未知字段'):
            self.frame.check(self.cfg)

    def test_unknown_registry_and_types_do_not_create_registries(self):
        for value in [{'type': 'Model'}, []]:
            with self.subTest(value=value):
                cfg = deepcopy(self.cfg)
                cfg['typo'] = value
                with self.assertRaisesRegex(ValueError, 'typo.*未注册'):
                    self.frame.check(cfg)
                self.assertIsNone(self.hub.get('typo'))
        self.hub['model']
        self.cfg['model'] = {'type': 'Missing'}
        with self.assertRaisesRegex(ValueError, 'model.type.*未注册'):
            self.frame.check(self.cfg)

    def test_missing_or_extra_constructor_arguments_fail_before_any_construction(self):
        created = []

        @self.hub.model.register('Model')
        def factory(required, *, optional=1):
            created.append(required)
            return required

        @self.hub.other.register('Other')
        def other():
            created.append('other')

        for params in [{}, {'required': 1, 'typo': 2}]:
            with self.subTest(params=params):
                cfg = {'other': {'type': 'Other'},
                       'model': {'type': 'Model', 'params': params}}
                with self.assertRaisesRegex(ValueError, 'model.params.*构造参数不匹配'):
                    self.builder.build_all_components(cfg)
                self.assertEqual(created, [])
                self.assertNotIn('other', self.context)

    def test_nested_forward_references_resolve_and_yaml_order_does_not_change_plan(self):
        @self.hub.node.register('Node')
        class Node:
            def __init__(self, value=None):
                self.value = value

        self.context['seed'] = 42
        cfg = {'node': [
            {'type': 'Node', 'name': 'consumer',
             'params': {'value': ['@producer', {'seed': '@seed',
                                               'tuple': ('@producer', '@seed')}]}},
            {'type': 'Node', 'name': 'independent'},
            {'type': 'Node', 'name': 'producer', 'params': {'value': 7}},
        ]}
        original = deepcopy(cfg)
        plan = self.builder.build_all_components(cfg)
        self.assertEqual(cfg, original)
        self.assertIs(self.context.consumer.value[0], self.context.producer)
        self.assertEqual(self.context.consumer.value[1]['seed'], 42)
        self.assertEqual(self.context.consumer.value[1]['tuple'], (self.context.producer, 42))
        names = [item.spec.name for item in plan.components]
        reversed_cfg = {'node': list(reversed(cfg['node']))}
        second_plan = self.builder.build_all_components(reversed_cfg)
        self.assertEqual(names, [item.spec.name for item in second_plan.components])

    def test_quoted_yaml_references_resolve_and_other_strings_remain_literal(self):
        self.hub.values.register('Values')(lambda **params: params)
        self.hub.consumer.register('Consumer')(lambda source: source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('''mate:
  seed: 42
  experiment: references
task:
  type: Task
consumer:
  type: Consumer
  params:
    source: "@values"
values:
  type: Values
  params:
    seed: '@seed'
    literal: _missing
    email: name@example.com
''', encoding='utf-8')
            cfg = self.frame.parse(path)
            original = deepcopy(cfg)
            plan = self.frame.check(cfg)
            self.frame.build_mate_task(cfg)
            self.frame.build_components(plan)
            self.assertIs(self.frame.context.consumer, self.frame.context.values)
            self.assertEqual(self.frame.context.values,
                             {'seed': 42, 'literal': '_missing', 'email': 'name@example.com'})
            self.assertEqual(cfg, original)

    def test_cycles_are_reported_before_construction(self):
        self.hub.node.register('Node')(lambda value: value)
        for nodes in [
            [{'type': 'Node', 'name': 'a', 'params': {'value': '@a'}}],
            [{'type': 'Node', 'name': 'a', 'params': {'value': '@b'}},
             {'type': 'Node', 'name': 'b', 'params': {'value': '@a'}}],
        ]:
            with self.subTest(nodes=nodes):
                with self.assertRaisesRegex(ValueError, '依赖成环.*a'):
                    self.builder.build_all_components({'node': nodes})
                self.assertNotIn('a', self.context)

    def test_variadic_factories_and_keyword_only_arguments_are_supported(self):
        self.hub.values.register('Values')(lambda **kwargs: kwargs)

        @self.hub.consumer.register('Consumer')
        def consumer(*, source):
            return source

        self.builder.build_all_components({
            'consumer': {'type': 'Consumer', 'params': {'source': '@values'}},
            'values': {'type': 'Values', 'params': {'free_key': 2}},
        })
        self.assertIs(self.context.consumer, self.context.values)
        self.assertEqual(self.context.values, {'free_key': 2})

    def test_duplicate_yaml_fields_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('mate:\n  seed: 42\n  seed: 2021\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '重复字段.*seed'):
                self.frame.parse(path)

    def test_recursive_yaml_values_are_rejected(self):
        cfg = deepcopy(self.cfg)
        params = {}
        params['cycle'] = params
        cfg['model'] = {'type': 'Model', 'params': params}
        with self.assertRaisesRegex(ValueError, '循环配置结构'):
            parse_config(cfg)

    def test_task_parameters_are_validated_and_passed_with_component_references(self):
        events = []
        self.hub.model.register('Model')(lambda: object())

        @self.hub.task.register('Parameterized')
        class ParameterizedTask:
            def __init__(self, context, hook_manager, model, label='default'):
                self.context, self.model, self.label = context, model, label

            def run(self):
                events.append((self.model is self.context.model, self.label))

        with tempfile.TemporaryDirectory() as directory:
            cfg = deepcopy(self.cfg)
            cfg['mate']['experiment_root'] = directory
            cfg['model'] = {'type': 'Model'}
            cfg['task'] = {'type': 'Parameterized', 'params': {'model': '@model', 'label': 'passed'}}
            self.frame.build_mate_task(cfg)
            self.frame.creat_experiment('unused', cfg)
            with patch('frame.frame.autodiscover_components'), redirect_stdout(io.StringIO()), \
                 redirect_stderr(io.StringIO()):
                self.frame._run(cfg)
            self.assertEqual(events, [(True, 'passed')])

        for params in [{}, {'model': '@missing'}, {'model': '@model', 'typo': 1},
                       {'context': '@model'}, {'hook_manager': '@model'}]:
            with self.subTest(params=params):
                cfg['task']['params'] = params
                with self.assertRaises(ValueError):
                    self.frame.check(cfg)

    def test_case_insensitive_duplicate_registration_is_rejected(self):
        registry = Registry('model')
        registry.register('CPMLP')(lambda: None)
        with self.assertRaisesRegex(ValueError, '重复注册'):
            registry.register('CpMlp')(lambda: None)


if __name__ == '__main__':
    unittest.main()
