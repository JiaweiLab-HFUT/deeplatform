from contextlib import nullcontext, redirect_stdout
import io
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from core.hook.hook import Hook
from core.hook.hook_manager import HookManager
from task.batteryLife_run import BatteryLifeRun


class CallbackHook(Hook):
    def __init__(self, event, callback, priority=0):
        super().__init__(priority)
        self.event, self.callback = event, callback

    def run(self, *args):
        self.callback(*args)


class TrainingHookControlTests(unittest.TestCase):
    def run_task(self, trigger_event=None, trigger_call=1, remote_stop=False, early_stop=False):
        """使用常量损失张量；模型、优化器和反向传播均为模拟对象。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        manager, events = HookManager(), []
        model, optimizer, scheduler = Mock(), Mock(), Mock()
        triggered = {'value': remote_stop}

        def set_trigger():
            triggered['value'] = True

        def check_trigger():
            value = triggered['value']
            triggered['value'] = False
            return value

        accelerator = SimpleNamespace(
            device='cuda:0', print=Mock(), backward=Mock(),
            prepare=lambda *args: args, accumulate=lambda model: nullcontext(),
            set_trigger=Mock(side_effect=set_trigger),
            check_trigger=Mock(side_effect=check_trigger),
        )
        dataset = MagicMock()
        dataset.__len__.return_value = 3
        batches = []
        for _ in range(3):
            batch = [MagicMock() for _ in range(7)]
            batch[5] = torch.ones(1, 1)
            batches.append(tuple(batch))
        datafactory = Mock()
        datafactory.get_dataset_loader.return_value = (dataset, batches)
        early_stop_component = SimpleNamespace(should_stop=early_stop, is_best=True, check=Mock())
        context = SimpleNamespace(
            __experiment_path__=temporary.name, epochs=3, device='cuda:0',
            accelerator=accelerator, model=model, datafactory=datafactory,
            optimizer=SimpleNamespace(build=Mock(return_value=optimizer)),
            scheduler=SimpleNamespace(build=Mock(return_value=scheduler)),
            loss=Mock(return_value=torch.tensor([[2.0]])), early_stop=early_stop_component,
            validate=SimpleNamespace(validate=Mock(return_value={'mape': 0.1})),
        )
        calls = []
        if trigger_event:
            def request_stop(context, *args):
                calls.append(trigger_event)
                if len(calls) == trigger_call:
                    context.accelerator.set_trigger()
            manager.register(CallbackHook(trigger_event, request_stop))
        manager.register(CallbackHook('epoch_over', lambda *args: events.append('epoch_over'), priority=10))
        manager.register(CallbackHook('after_train', lambda *args: events.append('after_train')))
        task = BatteryLifeRun(context, manager)
        with patch('task.batteryLife_run.joblib.dump'), \
             patch.object(task, 'log_epoch_result', side_effect=lambda *args: events.append('metrics')) as log, \
             patch.object(task, 'save_checkpoint', side_effect=lambda *args: events.append('checkpoint')) as save, \
             redirect_stdout(io.StringIO()):
            try:
                task.run()
            finally:
                manager.clear()
        return SimpleNamespace(manager=manager, events=events, optimizer=optimizer,
                               scheduler=scheduler, model=model, accelerator=accelerator,
                               context=context, log=log, save=save)

    def test_hook_trigger_finishes_current_epoch_then_stops(self):
        for event in ['before_train', 'before_epoch', 'before_iter', 'after_iter',
                      'after_epoch_train', 'epoch_over']:
            with self.subTest(event=event):
                result = self.run_task(event)
                self.assertEqual(result.optimizer.step.call_count, 3)
                self.assertEqual(result.scheduler.step.call_count, 3)
                self.assertEqual(result.context.validate.validate.call_count, 2)
                self.assertEqual(result.log.call_count, 1)
                self.assertEqual(result.log.call_args.args[2], {'loss': 2.0})
                self.assertTrue(result.accelerator.set_trigger.called)
                self.assertEqual(result.events, ['metrics', 'checkpoint', 'epoch_over', 'after_train'])

    def test_hook_can_trigger_after_multiple_epochs(self):
        result = self.run_task('epoch_over', trigger_call=2)
        self.assertEqual(result.optimizer.step.call_count, 6)
        self.assertEqual(result.log.call_count, 2)
        self.assertEqual(result.save.call_count, 2)
        self.assertEqual(result.events[-1], 'after_train')

    def test_no_stop_runs_all_epochs(self):
        result = self.run_task()
        self.assertEqual(result.optimizer.step.call_count, 9)
        self.assertEqual(result.log.call_count, 3)
        self.assertEqual(result.save.call_count, 3)
        self.assertEqual(result.events[-1], 'after_train')

    def test_remote_trigger_is_consumed_at_epoch_end(self):
        result = self.run_task(remote_stop=True)
        self.assertEqual(result.optimizer.step.call_count, 3)
        self.assertEqual(result.events, ['metrics', 'checkpoint', 'epoch_over', 'after_train'])

    def test_existing_early_stop_still_sets_accelerator_trigger(self):
        result = self.run_task(early_stop=True)
        result.accelerator.set_trigger.assert_called_once_with()
        self.assertEqual(result.optimizer.step.call_count, 3)
        self.assertEqual(result.events, ['metrics', 'checkpoint', 'epoch_over', 'after_train'])


if __name__ == '__main__':
    unittest.main()
