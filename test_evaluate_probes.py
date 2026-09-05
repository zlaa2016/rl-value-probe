import unittest

import numpy as np

from evaluate_probes import (
    activation_features,
    activation_lookup,
    heldout_predictions,
)


class EvaluationProbeTest(unittest.TestCase):
    def test_fixed_prompt_holdout_predicts_only_test_rollouts(self):
        rng = np.random.default_rng(0)
        prompts = np.repeat(["p0", "p1", "p2", "p3"], 2)
        X = rng.normal(size=(8, 5))
        y = X[:, 0] + 0.1 * rng.normal(size=8)

        predictions, train_mask, test_mask = heldout_predictions(
            X,
            y,
            prompts,
            train_prompts={"p0", "p1", "p2"},
            test_prompts={"p3"},
            alpha=1.0,
        )

        self.assertEqual(train_mask.sum(), 6)
        self.assertEqual(test_mask.sum(), 2)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(np.isfinite(predictions).all())

    def test_history_features_concatenate_prior_states(self):
        rows = [{
            "rollout_id": "r0",
            "model_stage": "base",
            "prompt_id": "p0",
            "reward": 1.0,
        }]
        activations = {
            "vectors": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "rollout_ids": np.asarray(["r0", "r0"]),
            "model_stages": np.asarray(["base", "base"]),
            "layers": np.asarray([8, 8]),
            "fractions": np.asarray([0.25, 0.50]),
        }
        lookup = activation_lookup(activations)

        X, _, _, _ = activation_features(
            rows,
            lookup,
            "base",
            8,
            [0.25, 0.50],
            0.50,
            "history",
        )

        np.testing.assert_allclose(X, [[1.0, 2.0, 3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
