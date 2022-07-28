import unittest
import torch
from model.layers import InverseDistanceAttention

class ModuleTest(unittest.TestCase):
    def test_invdist_attention(self):
        module = InverseDistanceAttention(n_features=3)

        x = torch.randint(5, (3, 5, 2))
        y = torch.randint(5, (3, 3, 2))

        attn_scores = module.compute_invdist_scores(x, y)

        true_scores = []
        for batch_idx in range(x.size(0)):
            batch = []
            for i in range(x.size(1)):
                t = []

                for j in range(y.size(1)):
                    dist = ((x[batch_idx, i, 0] - y[batch_idx, j, 0]).pow(2) +\
                            (x[batch_idx, i, 1] - y[batch_idx, j, 1]).pow(2)).sqrt()

                    t.append(dist)
            
                batch.append(t)

            true_scores.append(batch)

        true_scores = 1 / torch.tensor(true_scores)

        self.assertTrue(torch.equal(true_scores, attn_scores))

if __name__ == '__main__':
    # unittest.main()
    ModuleTest().test_invdist_attention()