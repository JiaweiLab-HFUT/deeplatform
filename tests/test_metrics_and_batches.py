import json
from types import SimpleNamespace
import unittest
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
            'weight': np.float64(1.), 'seen_unseen_id': 1,
        } for i in range(2)]
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            batch = batteryLife_collate_fn_baseline(samples)
        curves, masks, labels, _, _, weights, _ = batch
        self.assertEqual(curves.shape, (2, 2, 3, 4))
        self.assertEqual(labels.shape, (2, 1))
        self.assertEqual(weights.shape, (2,))
        self.assertEqual(labels.dtype, torch.float32)
        torch.testing.assert_close(labels, torch.tensor([[10.], [11.]]))
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
                 torch.zeros(2), torch.ones(2), torch.ones(2))
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


if __name__ == '__main__':
    unittest.main()
