import unittest

import torch
from torch.nn import functional as F

from models.gpt import GPT


class GPTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.model = GPT(vocab_size=16, block_size=8, n_layer=2, n_head=2, n_embd=16)

    def test_shapes_weight_tying_and_context_limit(self):
        tokens = torch.tensor([[1, 2, 3, 4], [2, 4, 6, 8]])
        embedded = self.model.embedding(tokens)
        logits = self.model(tokens)
        self.assertEqual(embedded.shape, (2, 4, 16))
        self.assertEqual(logits.shape, (2, 4, 16))
        torch.testing.assert_close(
            embedded[0, 1],
            self.model.embedding.token_embedding.weight[2]
            + self.model.embedding.position_embedding.weight[1],
        )
        self.assertIs(self.model.lm_head.weight, self.model.embedding.token_embedding.weight)
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 9), dtype=torch.long))

    @torch.no_grad()
    def test_future_tokens_do_not_affect_old_logits(self):
        self.model.eval()
        original = self.model(torch.tensor([[1, 2, 3, 4, 5]]))
        changed = self.model(torch.tensor([[1, 2, 3, 7, 8]]))
        torch.testing.assert_close(original[:, :3], changed[:, :3])
        torch.testing.assert_close(original, self.model(torch.tensor([[1, 2, 3, 4, 5]])))

    def test_model_can_overfit_one_batch(self):
        model = GPT(vocab_size=16, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        tokens = torch.tensor([[1, 5, 3, 9, 2, 0], [1, 5, 3, 9, 2, 0]])
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        for _ in range(70):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = model(inputs)
            final_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        self.assertLess(final_loss.item(), 0.1)
        self.assertTrue(torch.equal(logits.argmax(dim=-1), targets))


if __name__ == "__main__":
    unittest.main()
