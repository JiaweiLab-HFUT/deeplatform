import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import warnings

import numpy as np
import torch

from components.datafactory.BatteryLifeDataset import batteryLife_collate_fn_baseline
from components.metric.accuracy import Accuracy
from components.metric.mae import Mae
from components.metric.mape import Mape
from components.metric.mse import Mse
from components.metric.rmse import Rmse
from components.validate.batteryLife_validate import BatteryLifeValidate


class MetricAndBatchTests(unittest.TestCase):
    def test_metrics_return_python_floats(self):
        for metric in [Accuracy(0.15), Accuracy(0.10), Mae(), Mape(), Mse(), Rmse()]:
            with self.subTest(metric=metric):
                result = metric(np.array([10., 20.]), np.array([11., 19.]))
                self.assertIs(type(result), float)

    def test_accuracy_handles_zero_targets_without_dividing(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = Accuracy(0.15)(np.array([0., 0., 100., -100.]),
                                    np.array([0., 1., 110., -110.]))
        self.assertEqual(result, 75.0)

    def test_batch_stacks_numpy_labels_without_warnings_and_preserves_masks(self):
        samples = [{
            'cycle_curve_data': torch.ones(2, 3, 4),
            'curve_attn_mask': torch.tensor([1., 0.]),
            'labels': np.array([10. + i]),
            'life_class': 1, 'scaled_life_class': np.int64(0),
            'weight': np.float64(1. + i), 'seen_unseen_id': 1,
            'file_name': f'battery_{i}.pkl',
        } for i in range(2)]
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            batch = batteryLife_collate_fn_baseline(samples)
        curves, masks, labels, _, _, weights, _, names = batch
        self.assertEqual(names, ['battery_0.pkl', 'battery_1.pkl'])
        self.assertEqual(curves.shape, (2, 2, 3, 4))
        self.assertEqual(labels.shape, (2, 1))
        self.assertEqual(weights.shape, (2, 1))
        self.assertEqual(labels.dtype, torch.float32)
        torch.testing.assert_close(labels, torch.tensor([[10.], [11.]]))
        torch.testing.assert_close(weights, torch.tensor([[1.], [2.]]))
        predictions = torch.tensor([[11.], [14.]])
        weighted_errors = (predictions - labels).square() * weights
        self.assertEqual(weighted_errors.shape, (2, 1))
        self.assertEqual(weighted_errors.mean().item(), 9.5)
        self.assertEqual(curves[:, 1].count_nonzero().item(), 0)
        self.assertTrue(torch.all(curves[:, 0] == 1).item())
        self.assertTrue(torch.all(samples[0]['cycle_curve_data'] == 1).item())

    def test_validation_converts_tensor_metrics_and_skips_empty_groups(self):
        class Model(torch.nn.Module):
            def forward(self, values, mask):
                return values

        accelerator = SimpleNamespace(
            device=torch.device('cpu'), wait_for_everyone=lambda: None,
            gather_for_metrics=lambda values: values,
        )
        dataset = SimpleNamespace(label_scaler=SimpleNamespace(
            var_=np.array([1.]), mean_=np.array([0.]),
        ))
        values = torch.tensor([[10.], [20.]])
        batch = (values, torch.ones(2, 1), values, torch.ones(2),
                 torch.zeros(2), torch.ones(2), torch.ones(2), ['a.pkl', 'b.pkl'])
        validator = BatteryLifeValidate({
            'acc15': Accuracy(0.15), 'acc10': Accuracy(0.10),
            'custom': lambda y_true, y_pred: torch.tensor(0.25),
        })
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = validator.validate(accelerator, Model(), dataset, [batch], True)
        self.assertEqual(result['acc15'], 100.)
        for name, value in result.items():
            if name.startswith('unseen_'):
                self.assertIsNone(value)
            else:
                self.assertIs(type(value), float)
        self.assertIn('null', json.dumps(result, allow_nan=False))

    def test_prediction_records_preserve_names_windows_and_original_life_units(self):
        class Model(torch.nn.Module):
            def forward(self, values, mask):
                return values

        accelerator = SimpleNamespace(
            device=torch.device('cpu'), wait_for_everyone=lambda: None,
            gather_for_metrics=Mock(side_effect=lambda values, **kwargs: values),
        )
        dataset = SimpleNamespace(label_scaler=SimpleNamespace(var_=np.array([4.]), mean_=np.array([100.])))
        values = torch.tensor([[-1.], [1.], [0.5]])
        targets = torch.tensor([[-2.], [0.], [2.]])
        masks = torch.tensor([[1., 0.], [1., 1.], [1., 0.]])
        names = ['same.pkl', 'same.pkl', 'other.pkl']
        batches = [(values[part], masks[part], targets[part], torch.ones(len(values[part])),
                    torch.zeros(len(values[part])), torch.ones(len(values[part]), 1),
                    torch.ones(len(values[part])), names[part])
                   for part in (slice(0, 2), slice(2, 3))]
        validator = BatteryLifeValidate({'mae': Mae()})
        result, records = validator.validate(accelerator, Model(), dataset, batches,
                                             compute_seen_unseen=True, return_predictions=True)
        self.assertAlmostEqual(result['mae'], 7 / 3)
        self.assertEqual(records, [
            {'file_name': 'same.pkl', 'observed_cycles': 1, 'prediction': 98., 'target': 96.},
            {'file_name': 'same.pkl', 'observed_cycles': 2, 'prediction': 102., 'target': 100.},
            {'file_name': 'other.pkl', 'observed_cycles': 1, 'prediction': 101., 'target': 104.},
        ])
        self.assertNotIn('test_predictions', result)
        json.dumps(records, allow_nan=False)
        object_calls = [call for call in accelerator.gather_for_metrics.call_args_list
                        if call.kwargs.get('use_gather_object')]
        self.assertEqual([call.args[0] for call in object_calls], [names[:2], names[2:]])

    def test_prediction_records_use_gathered_filenames_and_trimmed_tensor_order(self):
        class Model(torch.nn.Module):
            def forward(self, values, mask):
                return values

        gathered = (torch.tensor([[10.], [20.], [30.]]), torch.tensor([[11.], [21.], [31.]]),
                    torch.ones(3), torch.tensor([1, 2, 3]))
        accelerator = SimpleNamespace(
            device=torch.device('cpu'), wait_for_everyone=lambda: None,
            gather_for_metrics=Mock(side_effect=[gathered, ['local.pkl', 'local.pkl', 'remote.pkl']]),
        )
        dataset = SimpleNamespace(label_scaler=SimpleNamespace(var_=np.array([1.]), mean_=np.array([0.])))
        batch = (torch.ones(2, 1), torch.ones(2, 3), torch.ones(2, 1), torch.ones(2),
                 torch.zeros(2), torch.ones(2, 1), torch.ones(2), ['local.pkl', 'local.pkl'])
        _, records = BatteryLifeValidate({'mae': Mae()}).validate(
            accelerator, Model(), dataset, [batch], return_predictions=True,
        )
        self.assertEqual(len(records), 3)
        self.assertEqual(records[-1], {'file_name': 'remote.pkl', 'observed_cycles': 3,
                                      'prediction': 30., 'target': 31.})
        accelerator.gather_for_metrics.side_effect = [gathered, ['missing.pkl']]
        with self.assertRaisesRegex(ValueError, '数量不一致'):
            BatteryLifeValidate({}).validate(accelerator, Model(), dataset, [batch], return_predictions=True)


if __name__ == '__main__':
    unittest.main()
