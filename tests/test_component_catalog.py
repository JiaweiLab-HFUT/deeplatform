from contextlib import redirect_stderr, redirect_stdout
from functools import partial
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from core.builder.builder import ComponentsBuilder
from core.config.schema import component_spec
from core.context.context import Context
from core.hook.hook import Hook
from core.register.catalog import (
    ComponentCatalogError, get_catalog, get_component_schema, list_scopes, list_types,
)
from core.register.registryhub import RegistryHub
from tools.component_catalog import main
from tools.component_upload import upload_component


class ComponentCatalogTests(unittest.TestCase):
    def setUp(self):
        self.hub = RegistryHub()
        self.options = {'register_hub': self.hub, 'discover': False}

    def schema(self, scope, type_name):
        return get_component_schema(scope, type_name, **self.options)

    def test_domains_types_sorting_case_insensitivity_and_read_only_errors(self):
        self.hub['empty']
        self.hub.model.register('Zulu')(lambda: None)
        self.hub.model.register('Alpha')(lambda: None)
        self.hub.loss.register('MSE')(lambda: None)
        self.assertEqual(list_scopes(**self.options), ['loss', 'model'])
        self.assertEqual(list_types('MODEL', **self.options), ['alpha', 'zulu'])
        self.assertEqual(self.schema('MODEL', 'ALPHA')['type'], 'alpha')
        for query, code in [(lambda: list_types('missing', **self.options), 'unknown_scope'),
                            (lambda: self.schema('model', 'missing'), 'unknown_type'),
                            (lambda: self.schema('model', ''), 'invalid_name')]:
            with self.subTest(code=code), self.assertRaises(ComponentCatalogError) as caught:
                query()
            self.assertEqual(caught.exception.code, code)
        self.assertIsNone(self.hub.get('missing'))

    def test_required_default_null_false_zero_and_keyword_only_parameters(self):
        @self.hub.model.register('Example')
        class Example:
            def __init__(self, size: int, width=64, enabled=False, ratio=0.0, optional=None,
                         *, labels: list[str] = ['a']):
                raise AssertionError('query must not instantiate a component')

        schema = self.schema('model', 'Example')
        fields = {field['name']: field for field in schema['params']}
        self.assertEqual(schema['required'], ['size'])
        self.assertEqual(schema['defaults'], {'width': 64, 'enabled': False, 'ratio': 0.0,
                                              'optional': None, 'labels': ['a']})
        self.assertTrue(fields['size']['required'])
        self.assertFalse(fields['size']['has_default'])
        self.assertNotIn('default', fields['size'])
        self.assertEqual(fields['size']['annotation'], 'int')
        self.assertEqual(fields['labels']['annotation'], 'list[str]')
        self.assertEqual(fields['labels']['kind'], 'KEYWORD_ONLY')
        self.assertTrue(fields['optional']['has_default'])
        self.assertIsNone(fields['optional']['default'])
        self.assertFalse(fields['optional']['required'])
        self.assertTrue(schema['configurable'])
        self.assertEqual(schema['issues'], [])
        json.dumps(schema, allow_nan=False)

    def test_defaults_are_detached_from_factory_and_other_returned_fields(self):
        default = {'names': ['a']}
        self.hub.model.register('Factory')(lambda options=default: None)
        first = self.schema('model', 'Factory')
        first['defaults']['options']['names'].append('defaults')
        self.assertEqual(first['params'][0]['default'], {'names': ['a']})
        first['params'][0]['default']['names'].append('params')
        second = self.schema('model', 'Factory')
        self.assertEqual(default, {'names': ['a']})
        self.assertEqual(second['defaults']['options'], default)

    def test_real_signature_of_inherited_classes_and_partial_factories(self):
        class Parent:
            def __init__(self, data, count=3):
                raise AssertionError('must not instantiate')

        @self.hub.model.register
        class Child(Parent):
            pass

        def factory(data, scale=2):
            raise AssertionError('must not call')

        self.hub.model.register('Partial')(partial(factory, scale=5))
        self.assertEqual(self.schema('model', 'Child')['required'], ['data'])
        self.assertEqual(self.schema('model', 'Child')['defaults'], {'count': 3})
        self.assertEqual(self.schema('model', 'Partial')['defaults'], {'scale': 5})

    def test_task_injection_is_hidden_and_matches_builder_validation(self):
        @self.hub.task.register('Custom')
        class Custom:
            def __init__(self, ctx, model, *, hook_manager, label='default'):
                raise AssertionError('must not instantiate')

        schema = self.schema('task', 'Custom')
        self.assertTrue(schema['configurable'], schema['issues'])
        self.assertEqual([field['name'] for field in schema['params']], ['model', 'label'])
        self.assertEqual(schema['required'], ['model'])
        self.assertEqual(schema['defaults'], {'label': 'default'})
        self.assertEqual([(item['name'], item['source']) for item in schema['injected_params']],
                         [('ctx', 'context'), ('hook_manager', 'hook_manager')])
        spec = component_spec({'type': 'Custom', 'params': {'model': 'reference', **schema['defaults']}},
                              'task', 'task', 'task', task=True)
        ComponentsBuilder(Context().bind('catalog'), self.hub).plan((), task=spec)

    def test_inherited_hooks_expose_constructor_instead_of_run_parameters(self):
        @self.hub.hook.register
        class Start(Hook):
            event = 'start'

            def run(self, context, result):
                raise AssertionError('must not run hook')

        schema = self.schema('hook', 'Start')
        self.assertEqual([field['name'] for field in schema['params']], ['priority'])
        self.assertEqual(schema['defaults'], {'priority': 0})
        self.assertTrue(schema['configurable'])
        self.hub.hook.register('Invalid')(lambda: None)
        self.assertFalse(self.schema('hook', 'Invalid')['configurable'])

    def test_variadic_and_positional_only_signatures_are_not_misrepresented(self):
        def flexible(required, *args, flag=True, **kwargs):
            pass

        def impossible(value, /):
            pass

        self.hub.model.register('Flexible')(flexible)
        self.hub.model.register('Impossible')(impossible)
        schema = self.schema('model', 'Flexible')
        self.assertEqual(schema['variadic'], {'args': 'args', 'kwargs': 'kwargs'})
        self.assertTrue(schema['accepts_extra_params'])
        self.assertEqual([field['name'] for field in schema['params']], ['required', 'flag'])
        self.assertEqual(schema['required'], ['required'])
        self.assertTrue(schema['configurable'])
        impossible_schema = self.schema('model', 'Impossible')
        self.assertFalse(impossible_schema['configurable'])
        self.assertFalse(impossible_schema['params'][0]['configurable'])
        self.assertTrue(impossible_schema['issues'])

    def test_unsupported_defaults_remain_optional_and_json_export_is_strict(self):
        cycle = []
        cycle.append(cycle)

        def factory(path=Path('/tmp'), dimensions=(1, 2), infinity=float('inf'),
                    nan=float('nan'), mapping={1: 'a'}, recursive=cycle, nullable=None):
            pass

        self.hub.model.register('Exotic')(factory)
        schema = self.schema('model', 'Exotic')
        self.assertEqual(schema['required'], [])
        self.assertEqual(schema['defaults'], {'nullable': None})
        for field in schema['params'][:-1]:
            self.assertTrue(field['has_default'])
            self.assertFalse(field['default_serializable'])
            self.assertFalse(field['required'])
            self.assertNotIn('default', field)
            self.assertIsInstance(field['default_repr'], str)
        json.dumps(get_catalog(**self.options), allow_nan=False)

    def test_invalid_factories_are_described_without_breaking_whole_catalog(self):
        self.hub.model.register('Object')(42)

        class Opaque:
            __signature__ = 'invalid signature'

            def __call__(self):
                raise AssertionError('must not call')

        self.hub.model.register('Opaque')(Opaque())
        self.hub.task.register('NoInjection')(lambda value: None)
        result = get_catalog(**self.options)
        self.assertEqual(result['schema_version'], 1)
        for scope in result['scopes']:
            for schema in scope['types']:
                self.assertFalse(schema['configurable'])
                self.assertTrue(schema['issues'])

    def test_discovery_failure_is_not_silently_treated_as_empty_catalog(self):
        with patch('core.register.catalog.autodiscover_components', side_effect=ImportError('broken module')):
            with self.assertRaisesRegex(ImportError, 'broken module'):
                list_scopes()
            output = io.StringIO()
            with redirect_stderr(output):
                self.assertEqual(main(['scopes']), 1)
            self.assertEqual(json.loads(output.getvalue())['error']['code'], 'discovery_failed')

    def test_annotations_are_reported_without_evaluating_forward_expressions(self):
        def factory(value: 'undefined_expression()'):
            pass

        self.hub.model.register('Future')(factory)
        schema = self.schema('model', 'Future')
        self.assertEqual(schema['params'][0]['annotation'], 'undefined_expression()')

    def test_cli_module_output_does_not_corrupt_json(self):
        def discover_and_list():
            print('module registration output')
            return ['model']

        output, errors = io.StringIO(), io.StringIO()
        with patch('tools.component_catalog.list_scopes', side_effect=discover_and_list), \
                redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(main(['scopes']), 0)
        self.assertEqual(json.loads(output.getvalue()), ['model'])
        self.assertIn('module registration output', errors.getvalue())

    def test_uploaded_component_appears_in_actual_cli_catalog(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'components').mkdir()
            (root / 'task').mkdir()
            source = root / 'example.py'
            source.write_text('''from core.register.registryhub import R
@R.model.register('CatalogExample')
class Example:
    def __init__(self, width, dropout=0.25):
        raise AssertionError('catalog must not construct components')
''')
            upload_component(source, 'model', project_root=root)
            environment = dict(os.environ)
            environment['PYTHONPATH'] = str(repo) + os.pathsep + environment.get('PYTHONPATH', '')
            result = subprocess.run([sys.executable, '-m', 'tools.component_catalog',
                                     'params', 'MODEL', 'CatalogExample'],
                                    cwd=root, env=environment, capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            schema = json.loads(result.stdout)
            self.assertEqual(schema['required'], ['width'])
            self.assertEqual(schema['defaults'], {'dropout': 0.25})
            self.assertTrue(schema['configurable'])
            self.assertEqual(schema['module'], 'components.uploaded.model.example')


if __name__ == '__main__':
    unittest.main()
