import unittest
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None
else:
    from extract_full_states import extract_full_generated_states


class FakeModel:
    config = SimpleNamespace(num_hidden_layers=4)

    def __init__(self):
        self.device = torch.device("cpu")

    def __call__(self, input_ids, **kwargs):
        batch, length = input_ids.shape
        base = torch.arange(batch * length * 3).reshape(batch, length, 3)
        hidden_states = tuple(base + 100 * layer for layer in range(5))
        return SimpleNamespace(hidden_states=hidden_states)


class FullStateExtractionTest(unittest.TestCase):
    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_extracts_only_generated_positions(self):
        payload = extract_full_generated_states(
            FakeModel(),
            prompt_ids=torch.tensor([10, 11]),
            generated_ids=torch.tensor([12, 13, 14]),
            layers=[1, 4],
        )

        self.assertEqual(payload["states"].shape, (2, 3, 3))
        np.testing.assert_array_equal(payload["layers"], [1, 4])
        np.testing.assert_array_equal(payload["token_ids"], [12, 13, 14])
        np.testing.assert_array_equal(payload["states"][0, 0], [106, 107, 108])


if __name__ == "__main__":
    unittest.main()
