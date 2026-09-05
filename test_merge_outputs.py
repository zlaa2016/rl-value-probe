from pathlib import Path
import tempfile
import unittest

import numpy as np

from merge_outputs import merge_output_directories
from run_mvp import load_outputs, save_outputs


def write_stage(directory, stage, rollout_id):
    save_outputs(
        directory / "rollouts.jsonl",
        directory / "activations.npz",
        rollout_records=[{
            "rollout_id": rollout_id,
            "model_stage": stage,
            "prompt_id": "p0",
            "rollout_index": 0,
            "reward": 1.0,
        }],
        activation_vectors=[np.array([1.0, 2.0])],
        activation_rollout_ids=[rollout_id],
        activation_model_stages=[stage],
        activation_layers=[8],
        activation_fractions=[0.25],
    )


class MergeOutputsTest(unittest.TestCase):
    def test_merges_distinct_model_stages(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            base_dir = root / "base"
            post_dir = root / "post"
            merged_dir = root / "merged"
            base_dir.mkdir()
            post_dir.mkdir()
            write_stage(base_dir, "base", "base-r0")
            write_stage(post_dir, "rlvr", "rlvr-r0")

            count = merge_output_directories(
                [base_dir, post_dir],
                merged_dir,
            )

            self.assertEqual(count, 2)
            loaded = load_outputs(
                merged_dir / "rollouts.jsonl",
                merged_dir / "activations.npz",
            )
            self.assertEqual(
                {row["model_stage"] for row in loaded[0]},
                {"base", "rlvr"},
            )
            self.assertEqual(len(loaded[1]), 2)

    def test_rejects_duplicate_stage_prompt_rollout_keys(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            write_stage(first, "base", "base-r0")
            write_stage(second, "base", "base-r1")

            with self.assertRaisesRegex(RuntimeError, "duplicate stage"):
                merge_output_directories(
                    [first, second],
                    root / "merged",
                )


if __name__ == "__main__":
    unittest.main()
