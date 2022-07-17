import unittest
import torch
from utils import metrics
from sklearn.metrics import r2_score


class MetricsTest(unittest.TestCase):
    def test_r2(self):
        y_pred = torch.rand(10)

        y_true = torch.rand(10)

        true_score = r2_score(y_true, y_pred)

        self.assertEqual(metrics.R2()(y_pred.unsqueeze(0), y_true.unsqueeze(0)), true_score)

    def test_mdape(self):
        y_pred = torch.rand((1,10))

        y_true = torch.rand((1, 10))

        self.assertIsInstance(metrics.MDAPE()(y_pred, y_true).item(), float)


if __name__ == '__main__':
    unittest.main()
