from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import yaml

from core.hook.hook import Hook
from core.hook.hook_manager import HookManager
from core.register.registryhub import RegistryHub
from core.task.Task import Task
from frame.frame import Frame


class RecordingHook(Hook):
    event = 'record'

    def __init__(self, records, label, priority=0):
        super().__init__(priority)
        self.records, self.label = records, label

    def run(self, value):
        self.records.append((self.label, value))


class HookManagerTests(unittest.TestCase):
    def test_instances_bind_explicitly_and_managers_are_isolated(self):
        records = []
        first, second = HookManager(), HookManager()
        hook = RecordingHook(records, 'first')
        self.assertEqual(first.emit('record', 0), [])
        first.register(hook)
        second.register(RecordingHook(records, 'second'))
        first.emit('record', 1)
        second.emit('record', 2)
        self.assertEqual(records, [('first', 1), ('second', 2)])

    def test_priority_and_registration_order_and_exact_unregistration(self):
        records, manager = [], HookManager()
        manager.register(RecordingHook(records, 'last', priority=10))
        token = manager.register(RecordingHook(records, 'remove', priority=0))
        manager.register(RecordingHook(records, 'first', priority=-1))
        manager.register(RecordingHook(records, 'equal_a', priority=0))
        manager.register(RecordingHook(records, 'equal_b', priority=0))
        self.assertTrue(manager.unregister(token))
        self.assertFalse(manager.unregister(token))
        manager.emit('record', 7)
        self.assertEqual(records, [('first', 7), ('equal_a', 7),
                                   ('equal_b', 7), ('last', 7)])

    def test_clear_releases_instances_and_old_tokens_do_not_remove_new_bindings(self):
        manager, records = HookManager(), []
        hook = RecordingHook(records, 'old')
        reference = weakref.ref(hook)
        token = manager.register(hook)
        del hook
        self.assertIsNotNone(reference())
        manager.clear()
        self.assertIsNone(reference())
        self.assertEqual(manager.emit('record', 0), [])
        manager.register(RecordingHook(records, 'new'))
        self.assertFalse(manager.unregister(token))
        manager.emit('record', 1)
        self.assertEqual(records, [('new', 1)])

    def test_handler_errors_are_logged_and_propagated(self):
        class FailingHook(Hook):
            event = 'record'

            def run(self, value):
                raise RuntimeError('handler failed')

        records, manager = [], HookManager()
        manager.register(FailingHook())
        manager.register(RecordingHook(records, 'next'))
        with self.assertLogs('core.hook.hook_manager', level='ERROR') as logs:
            with self.assertRaisesRegex(RuntimeError, 'handler failed'):
                manager.emit('record', 1)
        self.assertIn('handler failed', '\n'.join(logs.output))
        self.assertEqual(records, [])

    def test_binding_ownership_and_detachment(self):
        first, second = HookManager(), HookManager()
        hook = RecordingHook([], 'hook')
        with self.assertRaisesRegex(RuntimeError, '尚未绑定'):
            _ = hook.manager
        token = first.register(hook)
        self.assertIs(hook.manager, first)
        for manager in [first, second]:
            with self.assertRaisesRegex(ValueError, '只能绑定一次'):
                manager.register(hook)
        self.assertTrue(first.unregister(token))
        with self.assertRaisesRegex(RuntimeError, '解除绑定'):
            _ = hook.manager
        second.register(hook)
        self.assertIs(hook.manager, second)
        second.clear()
        with self.assertRaisesRegex(RuntimeError, '解除绑定'):
            _ = hook.manager


class HookExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / 'source.yaml'
        self.hub = RegistryHub()
        self.frame = Frame()
        self.frame.register_hub = self.hub
        self.records, self.managers, self.contexts = [], [], []
        records, managers, contexts = self.records, self.managers, self.contexts

        @self.hub.hook.register
        class CounterHook(Hook):
            event = 'record'

            def __init__(self, prefix='default', priority=0):
                super().__init__(priority)
                self.prefix, self.calls = prefix, 0

            def run(self, value):
                self.calls += 1
                records.append((self.prefix, value, self.calls))

        @self.hub.task.register
        class RecordingTask(Task):
            def __init__(self, context, hook_manager, value='passed', failure=None):
                super().__init__(context, hook_manager)
                self.value, self.failure = value, failure
                managers.append(hook_manager)
                contexts.append(context)
                if failure == 'constructor':
                    raise RuntimeError('task constructor failed')

            def run(self):
                self.hooks.emit('record', self.value)
                if self.failure == 'run':
                    raise RuntimeError('task run failed')

        self.config = {
            'mate': {'seed': 42, 'experiment': 'hooks',
                     'experiment_root': str(Path(self.temporary.name) / 'experiment')},
            'task': {'type': 'RecordingTask'},
            'hook': [{'type': 'CounterHook', 'name': 'counter',
                      'params': {'prefix': 'configured'}}],
        }

    def run_experiment(self):
        self.source.write_text(yaml.safe_dump(self.config), encoding='utf-8')
        with patch('frame.frame.autodiscover_components'), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return self.frame.run(self.source)

    def test_reusing_frame_creates_fresh_instances_and_disabled_hooks_do_not_execute(self):
        first = self.run_experiment()
        second = self.run_experiment()
        del self.config['hook']
        self.run_experiment()
        self.assertNotEqual(first, second)
        self.assertEqual(self.records, [('configured', 'passed', 1),
                                        ('configured', 'passed', 1)])
        self.assertEqual(len({id(manager) for manager in self.managers}), 3)
        self.assertEqual(len({id(context) for context in self.contexts}), 3)
        self.assertNotIn('counter', self.frame.context)
        self.assertEqual(list(self.hub.hook.keys()), ['counterhook'])
        for manager, context in zip(self.managers, self.contexts):
            self.assertEqual(manager.emit('record', 'after_run'), [])
            self.assertNotIn('hook_manager', context)

    def test_manager_is_cleared_after_task_constructor_or_run_failure(self):
        for failure in ['constructor', 'run']:
            with self.subTest(failure=failure):
                self.config['task']['params'] = {'failure': failure}
                with self.assertRaisesRegex(RuntimeError, f'task {failure} failed'):
                    self.run_experiment()
                manager = self.managers[-1]
                self.assertEqual(manager.emit('record', 'after_failure'), [])
                status_path = Path(self.frame.context.__experiment_path__) / 'status.json'
                self.assertEqual(json.loads(status_path.read_text())['status'], 'failed')
                with self.assertRaises(RuntimeError):
                    _ = self.frame.context.counter.manager

    def test_hook_manager_configuration_override_is_rejected(self):
        self.config['task']['params'] = {'hook_manager': '@counter'}
        with self.assertRaisesRegex(ValueError, 'task.params.hook_manager.*框架注入'):
            self.frame.check(self.config)

    def test_task_must_accept_explicit_manager_before_any_components_are_created(self):
        self.hub.task.register('OldTask')(lambda context: context)
        self.config['task'] = {'type': 'OldTask'}
        with patch.object(self.frame, 'build_components') as build:
            with self.assertRaisesRegex(ValueError, 'task.params.*hook_manager'):
                self.run_experiment()
            build.assert_not_called()

    def test_invalid_hook_types_and_missing_events_fail_before_construction(self):
        class MissingEvent(Hook):
            def run(self): ...

        class MissingRun(Hook):
            event = 'record'

        for name, factory in [('Function', lambda: object()),
                              ('MissingEvent', MissingEvent), ('MissingRun', MissingRun)]:
            self.hub.hook.register(name)(factory)
            self.config['hook'] = {'type': name}
            with self.subTest(name=name), patch.object(self.frame, 'build_components') as build:
                with self.assertRaisesRegex(ValueError, 'hook.type.*Hook'):
                    self.run_experiment()
                build.assert_not_called()


if __name__ == '__main__':
    unittest.main()
