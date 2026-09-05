import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_mvp import load_outputs, rollout_key, save_outputs


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

            loaded = load_outputs(rollout_path, activation_path)
            self.assertEqual(loaded[0][0]["rollout_id"], "r1")
            np.testing.assert_allclose(loaded[1], [[1.0, 2.0]])
            self.assertEqual(loaded[2], ["r1"])

    def test_resume_key_uses_stage_prompt_and_rollout_index(self):
        row = {"custom_id": "prompt-7"}
        self.assertEqual(
            rollout_key("base", row, 0, 3),
            ("base", "prompt-7", 3),
        )

    def test_resume_requires_both_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rollout_path = directory / "rollouts.jsonl"
            rollout_path.write_text('{"rollout_id": "r1"}\n')

            with self.assertRaisesRegex(RuntimeError, "only one checkpoint"):
                load_outputs(rollout_path, directory / "activations.npz")


if __name__ == "__main__":
    unittest.main()
