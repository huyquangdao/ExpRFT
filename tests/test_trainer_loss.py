import unittest

import torch
import torch.nn.functional as F

from exprft.trainer import WeightedCollator, dataset_token_normalizer, weighted_response_loss


def _standard_sft_loss(logits, labels):
    sl = logits[:, :-1, :].reshape(-1, logits.size(-1)).float()
    ll = labels[:, 1:].reshape(-1)
    return F.cross_entropy(sl, ll, ignore_index=-100)


class TestWeightedResponseLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.T, self.V = 3, 7, 11
        self.logits = torch.randn(self.B, self.T, self.V)
        self.labels = torch.randint(0, self.V, (self.B, self.T))
        self.labels[0, :3] = -100          # prompt tokens
        self.labels[1, :5] = -100
        self.labels[2, :2] = -100
        self.labels[2, 6] = -100           # padding
        self.n_tokens = int((self.labels[:, 1:] != -100).sum())

    def test_unit_weights_with_token_count_equal_standard_sft(self):
        loss = weighted_response_loss(self.logits, self.labels, torch.ones(self.B), self.n_tokens)
        self.assertAlmostEqual(loss.item(), _standard_sft_loss(self.logits, self.labels).item(), places=5)

    def test_additive_over_any_micro_batching(self):
        """The batch loss is the sum of the micro-batch losses."""
        w = torch.tensor([0.3, 2.0, 1.1])
        c = 17.0
        full = weighted_response_loss(self.logits, self.labels, w, c)
        parts = sum(weighted_response_loss(self.logits[i:i + 1], self.labels[i:i + 1], w[i:i + 1], c)
                    for i in range(self.B))
        self.assertAlmostEqual(full.item(), parts.item(), places=5)

    def test_batch_size_one_keeps_the_weight(self):
        c = 5.0
        a = weighted_response_loss(self.logits[:1], self.labels[:1], torch.tensor([1.0]), c)
        b = weighted_response_loss(self.logits[:1], self.labels[:1], torch.tensor([2.5]), c)
        self.assertAlmostEqual(b.item(), 2.5 * a.item(), places=5)

    def test_zero_weight_rows_do_not_contribute(self):
        c = 9.0
        loss = weighted_response_loss(self.logits, self.labels, torch.tensor([1.0, 0.0, 1.0]), c)
        sub = weighted_response_loss(self.logits[[0, 2]], self.labels[[0, 2]], torch.ones(2), c)
        self.assertAlmostEqual(loss.item(), sub.item(), places=5)

    def test_masked_positions_ignored(self):
        logits2 = self.logits.clone()
        logits2[:, 0, :] += 100.0          # position 0 predicts label[1]; masked in every row
        a = weighted_response_loss(self.logits, self.labels, torch.ones(self.B), 3.0)
        b = weighted_response_loss(logits2, self.labels, torch.ones(self.B), 3.0)
        self.assertAlmostEqual(a.item(), b.item(), places=5)

    def test_normalizer_must_be_positive(self):
        with self.assertRaises(ValueError):
            weighted_response_loss(self.logits, self.labels, torch.ones(self.B), 0.0)

    def test_dataset_token_normalizer(self):
        # mean(w*n) = (1*10 + 3*20) / 2 = 35 ; times batch 4 = 140
        self.assertAlmostEqual(dataset_token_normalizer([1.0, 3.0], [10, 20], 4), 140.0)
        with self.assertRaises(ValueError):
            dataset_token_normalizer([], [], 4)


class TestCollator(unittest.TestCase):
    def test_padding_and_masks(self):
        col = WeightedCollator(pad_token_id=99)
        batch = col([
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "weight": 0.5},
            {"input_ids": [4, 5], "labels": [-100, 5], "weight": 2.0},
        ])
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3], [4, 5, 99]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2, 3], [-100, 5, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(batch["weight"].tolist(), [0.5, 2.0])


if __name__ == "__main__":
    unittest.main()
