import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_mvp import save_outputs


class IncrementalOutputCheckpointTest(unittest.TestCase):
    def test_saves_matching_rollout_and_activation_files(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rollout_path = directory / "rollouts.jsonl"
            activation_path = directory / "activations.npz"

            save_outputs(
                rollout_path,
                activation_path,
                rollout_records=[{"rollout_id": "r1", "reward": 1.0}],
                activation_vectors=[np.array([1.0, 2.0])],
                activation_rollout_ids=["r1"],
                activation_model_stages=["base"],
                activation_layers=[8],
                activation_fractions=[0.25],
            )

            with open(rollout_path) as f:
                self.assertEqual(json.loads(f.readline())["rollout_id"], "r1")
            activations = np.load(activation_path)
            np.testing.assert_allclose(activations["vectors"], [[1.0, 2.0]])
            self.assertEqual(activations["rollout_ids"].tolist(), ["r1"])


if __name__ == "__main__":
    unittest.main()
