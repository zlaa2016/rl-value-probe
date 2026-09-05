import unittest

import numpy as np

from analysis_utils import regression_metrics
from probe import evaluate_ridge_with_label_shuffle, paired_activation_deltas


class LabelShuffleControlTest(unittest.TestCase):
    def test_constant_targets_make_r2_and_rank_correlation_undefined(self):
        metrics = regression_metrics(
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.1, 0.2, 0.3]),
        )

        self.assertTrue(np.isnan(metrics["r2"]))
        self.assertTrue(np.isnan(metrics["spearman_like"]))
        self.assertEqual(metrics["target_variance"], 0.0)
        self.assertAlmostEqual(metrics["mse"], 0.14 / 3)

    def test_rank_correlation_uses_average_ranks(self):
        metrics = regression_metrics(
            np.asarray([0.0, 0.0, 1.0, 1.0]),
            np.asarray([0.0, 0.0, 1.0, 1.0]),
        )

        self.assertAlmostEqual(metrics["spearman_like"], 1.0)

    def test_real_signal_beats_shuffled_training_labels(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(160, 8))
        direction = rng.normal(size=8)
        y = X @ direction + rng.normal(scale=0.05, size=160)

        metrics = evaluate_ridge_with_label_shuffle(
            X[:120],
            y[:120],
            X[120:],
            y[120:],
            n_shuffles=30,
            rng=np.random.default_rng(11),
        )

        self.assertGreater(metrics["r2"], 0.9)
        self.assertGreater(metrics["r2"], metrics["r2_null_mean"])
        self.assertEqual(metrics["label_shuffles"], 30)
        self.assertLessEqual(metrics["r2_permutation_p"], 2 / 31)

    def test_activation_deltas_are_paired_by_rollout_id(self):
        acts = {
            "vectors": np.array([[1.0], [10.0], [13.0], [3.0]]),
            "rollout_ids": np.array(["a", "b", "b", "a"]),
            "model_stages": np.array(["base"] * 4),
            "layers": np.array([8] * 4),
            "fractions": np.array([0.25, 0.25, 0.50, 0.50]),
        }

        deltas, rollout_ids = paired_activation_deltas(
            acts,
            stage="base",
            layer=8,
            start_fraction=0.25,
            end_fraction=0.50,
        )

        np.testing.assert_array_equal(rollout_ids, ["a", "b"])
        np.testing.assert_allclose(deltas[:, 0], [2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
