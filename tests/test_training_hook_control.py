from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from core.hook.hook import Hook
from core.hook.hook_manager import HookManager
from components.others.earlystop import EarlyStop
from task.batteryLife_run import BatteryLifeRun


class CallbackHook(Hook):
    def __init__(self, event, callback, priority=0):
        super().__init__(priority)
        self.event, self.callback = event, callback

    def run(self, *args):
        self.callback(*args)


class TrainingHookControlTests(unittest.TestCase):
    def run_task(self, trigger_event=None, trigger_call=1, remote_stop=False, early_stop=False,
                 validation_scores=None, is_main_process=True):
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
            is_main_process=is_main_process,
        )
        dataset = MagicMock()
        dataset.__len__.return_value = 3
        batches = []
        for _ in range(3):
            batch = [MagicMock() for _ in range(7)]
            batch[5] = torch.ones(1, 1)
            batch.append(['battery.pkl'])
            batches.append(tuple(batch))
        datafactory = Mock()
        datafactory.get_dataset_loader.return_value = (dataset, batches)
        early_stop_component = SimpleNamespace(should_stop=early_stop, is_best=True, check=Mock())
        if validation_scores is not None:
            early_stop_component = EarlyStop(False, 5, 0, 'mape', 'less', verbose=False)
        evaluation = {'epoch': 0}

        def validate(*args, return_predictions=False, **kwargs):
            epoch = evaluation['epoch']
            if not return_predictions:
                return {'mape': 0.1 if validation_scores is None else validation_scores[epoch]}
            evaluation['epoch'] += 1
            return {'mape': 0.5 + epoch}, [
                {'file_name': 'battery.pkl', 'observed_cycles': 1,
                 'prediction': 100. + epoch, 'target': 110.},
                {'file_name': 'battery.pkl', 'observed_cycles': 10,
                 'prediction': 102. + epoch, 'target': 110.},
            ]
        context = SimpleNamespace(
            __experiment_path__=temporary.name, epochs=3, device='cuda:0',
            accelerator=accelerator, model=model, datafactory=datafactory,
            optimizer=SimpleNamespace(build=Mock(return_value=optimizer)),
            scheduler=SimpleNamespace(build=Mock(return_value=scheduler)),
            loss=Mock(return_value=torch.tensor([[2.0]])), early_stop=early_stop_component,
            validate=SimpleNamespace(validate=Mock(side_effect=validate)),
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
        original_log = task.log_epoch_result

        def log_epoch(*args):
            events.append('metrics')
            original_log(*args)

        with patch('task.batteryLife_run.joblib.dump'), \
             patch.object(task, 'log_epoch_result', side_effect=log_epoch) as log, \
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

    def test_best_epoch_predictions_are_saved_but_not_in_epoch_logs(self):
        result = self.run_task(validation_scores=[0.3, 0.1, 0.2])
        root = Path(result.context.__experiment_path__)
        best = json.loads((root / 'best_result.json').read_text())
        self.assertEqual(best['epoch'], 2)
        self.assertEqual(best['validation'], {'mape': 0.1})
        self.assertEqual(best['test'], {'mape': 1.5})
        self.assertEqual(best['test_predictions_aggregation'], 'mean_by_battery')
        self.assertEqual(best['test_predictions'], [
            {'file_name': 'battery.pkl', 'prediction': 102., 'target': 110.},
        ])
        self.assertEqual(result.save.call_count, 2)
        metrics = (root / 'metrics.jsonl').read_text()
        self.assertEqual(len(metrics.splitlines()), 3)
        self.assertNotIn('file_name', metrics)
        self.assertNotIn('test_predictions', metrics)
        self.assertNotIn('battery.pkl', str(result.accelerator.print.call_args_list))
        self.assertFalse((root / 'best_result.json.tmp').exists())

    def test_only_main_process_writes_best_result_and_epoch_metrics(self):
        result = self.run_task(validation_scores=[0.3, 0.1, 0.2], is_main_process=False)
        root = Path(result.context.__experiment_path__)
        self.assertFalse((root / 'best_result.json').exists())
        self.assertFalse((root / 'metrics.jsonl').exists())

    def test_battery_average_groups_nonadjacent_windows_with_equal_weight(self):
        predictions = [
            {'file_name': 'b.pkl', 'observed_cycles': 1, 'prediction': 10., 'target': 25.},
            {'file_name': 'a.pkl', 'observed_cycles': 2, 'prediction': 40., 'target': 50.},
            {'file_name': 'b.pkl', 'observed_cycles': 100, 'prediction': 30., 'target': 25.},
        ]
        self.assertEqual(BatteryLifeRun.average_battery_predictions(predictions), [
            {'file_name': 'b.pkl', 'prediction': 20., 'target': 25.},
            {'file_name': 'a.pkl', 'prediction': 40., 'target': 50.},
        ])
        self.assertEqual(BatteryLifeRun.average_battery_predictions([]), [])
        self.assertEqual(len(predictions), 3)
        self.assertEqual(predictions[0]['prediction'], 10.)

    def test_battery_average_rejects_inconsistent_life_labels(self):
        predictions = [
            {'file_name': 'same.pkl', 'prediction': 10., 'target': 20.},
            {'file_name': 'same.pkl', 'prediction': 30., 'target': 40.},
        ]
        with self.assertRaisesRegex(ValueError, '真实寿命不一致'):
            BatteryLifeRun.average_battery_predictions(predictions)


if __name__ == '__main__':
    unittest.main()
