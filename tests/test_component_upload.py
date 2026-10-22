from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.component_upload import (
    ComponentStore, ComponentUploadError, list_components, upload_component,
)


SOURCE = b'''from core.register.registryhub import R

@R.model.register("Example")
class Example:
    def __init__(self, value=1):
        self.value = value
'''


class ProjectTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'components').mkdir()
        (self.root / 'task').mkdir()
        self.store = ComponentStore(self.root)

    def assert_upload_error(self, code, scope, filename, source):
        with self.assertRaises(ComponentUploadError) as caught:
            self.store.upload(scope, filename, source)
        self.assertEqual(caught.exception.code, code)
        return caught.exception


class ComponentStoreTests(ProjectTestCase):
    def test_upload_is_persistent_and_listed_without_executing_source(self):
        marker = self.root / 'executed'
        source = SOURCE + f'\nopen({str(marker)!r}, "w").write("executed")\n'.encode()
        result = self.store.upload('MODEL', 'example.py', source)
        target = self.root / result['path']
        self.assertEqual(target.read_bytes(), source)
        self.assertEqual(result['sha256'], hashlib.sha256(source).hexdigest())
        self.assertEqual(result['activation'], 'next_experiment')
        self.assertEqual(result['components'][0]['module'], 'components.uploaded.model.example')
        self.assertEqual(ComponentStore(self.root).list_components(), result['components'])
        self.assertFalse(marker.exists())

    def test_existing_type_is_detected_case_insensitively_without_import(self):
        original = SOURCE + b'\nraise RuntimeError("do not import during upload")\n'
        path = self.root / 'components' / 'builtin.py'
        path.write_bytes(original)
        self.assert_upload_error('type_exists', 'model', 'other.py', SOURCE.replace(b'Example', b'EXAMPLE'))
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.root / 'components/uploaded/model/other.py').exists())

    def test_existing_filename_cannot_be_overwritten(self):
        self.store.upload('model', 'example.py', SOURCE)
        self.assert_upload_error('file_exists', 'model', 'example.py', SOURCE.replace(b'Example', b'Another'))
        self.assertEqual((self.root / 'components/uploaded/model/example.py').read_bytes(), SOURCE)

    def test_invalid_names_and_paths_cannot_write_components(self):
        for scope, filename in [('model', '../outside.py'), ('../model', 'model.py'),
                                ('model', 'bad\\file.py'), ('model', '__init__.py'),
                                ('model', '_private.py'), ('model', 'class.py'),
                                ('model', 'example.zip'), ('mate', 'example.py')]:
            with self.subTest(scope=scope, filename=filename), self.assertRaises(ComponentUploadError):
                self.store.upload(scope, filename, SOURCE)
        self.assertEqual(list(self.root.rglob('*.py')), [])

    def test_invalid_source_or_registration_leaves_no_python_file(self):
        cases = [('empty_file', b''), ('invalid_source', b'def invalid(:'),
                 ('invalid_source', b'return 1'), ('invalid_source', b'\xff'),
                 ('invalid_source', b'# coding: latin-1\n' + SOURCE),
                 ('missing_registration', b'class Example: pass'),
                 ('invalid_registration', SOURCE.replace(b'"Example"', b'NAME')),
                 ('scope_mismatch', SOURCE.replace(b'R.model', b'R.metric')),
                 ('duplicate_registration', SOURCE + SOURCE.replace(b'Example', b'EXAMPLE'))]
        for code, source in cases:
            with self.subTest(code=code):
                self.assert_upload_error(code, 'model', 'invalid.py', source)
        self.assertEqual(list(self.root.rglob('*.py')), [])

    def test_aliases_subscripts_and_multiple_types_in_one_module(self):
        source = b'''from core.register.registryhub import R as hub
@hub["MODEL"].register
class First: pass
@hub.model.register(key="SecondAlias")
def second_factory(): return object()
'''
        result = self.store.upload('model', 'multiple.py', source)
        self.assertEqual([item['type'] for item in result['components']], ['first', 'secondalias'])

    def test_tasks_are_stored_under_components(self):
        result = self.store.upload('task', 'example.py', SOURCE.replace(b'R.model', b'R.task'))
        self.assertEqual(result['path'], 'components/uploaded/task/example.py')
        self.assertEqual(result['components'][0]['module'], 'components.uploaded.task.example')

    def test_concurrent_uploads_cannot_install_the_same_type_twice(self):
        def upload(index):
            try:
                ComponentStore(self.root).upload('model', f'example{index}.py', SOURCE)
                return 'installed'
            except ComponentUploadError as error:
                return error.code

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(upload, range(2)))
        self.assertEqual(sorted(results), ['installed', 'type_exists'])
        self.assertEqual(len(self.store.list_components()), 1)

    def test_upload_directory_symlinks_are_rejected(self):
        outside = self.root / 'outside'
        outside.mkdir()
        uploaded = self.root / 'components/uploaded'
        uploaded.mkdir()
        (uploaded / 'model').symlink_to(outside, target_is_directory=True)
        self.assert_upload_error('invalid_path', 'model', 'example.py', SOURCE)
        self.assertEqual(list(outside.iterdir()), [])

    def test_failed_publish_removes_temporary_source(self):
        with patch('tools.component_upload.os.link', side_effect=OSError('publish failed')):
            with self.assertRaisesRegex(OSError, 'publish failed'):
                self.store.upload('model', 'example.py', SOURCE)
        self.assertEqual(list(self.root.rglob('*.py')), [])
        self.assertEqual(list(self.root.rglob('*.tmp')), [])


class ComponentFileTests(ProjectTestCase):
    def write_source(self, content=SOURCE, filename='example.py'):
        path = self.root / filename
        path.write_bytes(content)
        return path

    def test_file_function_preserves_source_and_lists_installed_component(self):
        source = self.write_source()
        result = upload_component(source, 'MODEL', project_root=self.root)
        self.assertEqual(result['path'], 'components/uploaded/model/example.py')
        self.assertEqual((self.root / result['path']).read_bytes(), SOURCE)
        self.assertEqual(source.read_bytes(), SOURCE)
        self.assertEqual(list_components('model', project_root=self.root), result['components'])
        self.assertEqual(list_components('metric', project_root=self.root), [])

    def test_filename_override_does_not_execute_source(self):
        marker = self.root / 'executed'
        content = SOURCE + f'\nopen({str(marker)!r}, "w").write("executed")\n'.encode()
        source = self.write_source(content)
        result = upload_component(source, 'model', filename='renamed.py', project_root=self.root)
        self.assertEqual(result['path'], 'components/uploaded/model/renamed.py')
        self.assertFalse(marker.exists())
        with self.assertRaises(ComponentUploadError) as caught:
            upload_component(source, 'model', filename='renamed.py', project_root=self.root)
        self.assertEqual(caught.exception.code, 'file_exists')
        self.assertEqual(source.read_bytes(), content)

    def test_missing_and_oversized_files_do_not_install_source(self):
        with self.assertRaises(FileNotFoundError):
            upload_component(self.root / 'missing.py', 'model', project_root=self.root)
        source = self.write_source()
        with self.assertRaises(ComponentUploadError) as caught:
            upload_component(source, 'model', project_root=self.root, max_upload_bytes=10)
        self.assertEqual(caught.exception.code, 'file_too_large')
        self.assertEqual(list_components(project_root=self.root), [])

    def test_uploaded_component_is_discovered_and_built_in_a_fresh_process(self):
        marker = self.root / 'imported.txt'
        source = self.write_source(SOURCE + f'\nopen({str(marker)!r}, "w").write("imported")\n'.encode())
        upload_component(source, 'model', project_root=self.root)
        self.assertFalse(marker.exists())
        script = '''from core.register.autodiscover import autodiscover_components
from core.builder.builder import ComponentsBuilder
from core.context.context import Context
autodiscover_components(('components.uploaded',))
context = Context().bind('test')
ComponentsBuilder(context).build_all_components({'model': {'type': 'Example', 'params': {'value': 7}}})
assert context.model.value == 7
print('discovered and built')
'''
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get('PYTHONPATH', '')
        result = subprocess.run([sys.executable, '-c', script], cwd=self.root, env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('discovered and built', result.stdout)
        self.assertEqual(marker.read_text(), 'imported')


class ComponentCommandTests(ProjectTestCase):
    def command(self, *args):
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get('PYTHONPATH', '')
        return subprocess.run([sys.executable, '-m', 'tools.component_upload', *map(str, args),
                               '--project-root', str(self.root)], cwd=self.root, env=environment,
                              capture_output=True, text=True, timeout=30)

    def test_command_upload_list_and_duplicate_errors(self):
        source = self.root / 'example.py'
        source.write_bytes(SOURCE)
        result = self.command('upload', source, '--scope', 'model', '--filename', 'renamed.py')
        self.assertEqual(result.returncode, 0, result.stderr)
        installed = json.loads(result.stdout)
        self.assertEqual(installed['path'], 'components/uploaded/model/renamed.py')
        listing = self.command('list', '--scope', 'MODEL')
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertEqual(json.loads(listing.stdout), {'count': 1, 'items': installed['components']})
        conflict = self.command('upload', source, '--scope', 'model')
        self.assertEqual(conflict.returncode, 1)
        self.assertEqual(json.loads(conflict.stderr)['error']['code'], 'type_exists')
        self.assertEqual(conflict.stdout, '')
        self.assertEqual(source.read_bytes(), SOURCE)

    def test_command_validation_and_io_failures_return_nonzero(self):
        source = self.root / 'example.py'
        source.write_bytes(SOURCE)
        cases = [([source, '--scope', 'metric'], 'scope_mismatch'),
                 ([source, '--scope', 'model', '--max-upload-bytes', '10'], 'file_too_large'),
                 ([self.root / 'missing.py', '--scope', 'model'], 'invalid_input')]
        for arguments, code in cases:
            with self.subTest(code=code):
                result = self.command('upload', *arguments)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(json.loads(result.stderr)['error']['code'], code)
        self.assertEqual(self.command('upload', source).returncode, 2)
        self.assertEqual(list_components(project_root=self.root), [])


if __name__ == '__main__':
    unittest.main()
