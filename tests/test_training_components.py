import unittest

import torch
import yaml

from components.loss.mse import MSELoss
from components.others.earlystop import EarlyStop


class TrainingComponentTests(unittest.TestCase):
    def test_sample_weights_apply_to_each_sample_and_gradients(self):
        predictions = torch.tensor([[1.0], [3.0]], requires_grad=True)
        loss = MSELoss()(predictions, torch.zeros_like(predictions), torch.tensor([1.0, 2.0]))
        self.assertEqual(loss.item(), 9.5)
        loss.backward()
        torch.testing.assert_close(predictions.grad, torch.tensor([[1.0], [6.0]]))

    def test_early_stop_direction_from_yaml(self):
        direction = yaml.safe_load('direction: less')['direction']
        early_stop = EarlyStop(True, 2, 0, 'mape', direction, verbose=False)
        early_stop.check({'mape': 0.3}, 1)
        self.assertTrue(early_stop.is_best)
        early_stop.check({'mape': 0.4}, 2)
        early_stop.check({'mape': 0.5}, 3)
        self.assertTrue(early_stop.should_stop)


if __name__ == '__main__':
    unittest.main()
