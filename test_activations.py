import unittest

import numpy as np
import torch

from activations import _token_confidence_signals


class _IdentityLMHeadModel:
    @staticmethod
    def lm_head(hidden):
        return hidden


class TokenConfidenceSignalsTest(unittest.TestCase):
    def test_chosen_tokens_are_gathered_from_lm_head_logits(self):
        hidden = torch.tensor([
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0],
        ])
        generated_ids = torch.tensor([0, 2])

        signals = _token_confidence_signals(
            _IdentityLMHeadModel(),
            hidden,
            generated_ids,
            chunk_size=1,
        )

        expected = torch.log_softmax(hidden, dim=-1)[[0, 1], [0, 2]].numpy()
        np.testing.assert_allclose(signals["token_logprob"], expected)
        self.assertEqual(signals["token_entropy"].shape, (2,))
        self.assertEqual(signals["top1_prob"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
