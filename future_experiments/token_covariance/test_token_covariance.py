import unittest

import numpy as np

from token_covariance import (
    feature_covariance,
    prefix_summaries,
    spectral_summary,
    token_covariance,
    token_gram,
)


class TokenCovarianceTest(unittest.TestCase):
    def setUp(self):
        self.states = np.asarray([
            [1.0, 2.0, 4.0],
            [2.0, 1.0, 3.0],
            [4.0, 0.0, 1.0],
            [5.0, 1.0, 0.0],
        ])

    def test_matrix_shapes_and_symmetry(self):
        feature = feature_covariance(self.states)
        gram = token_gram(self.states)
        token_cov = token_covariance(self.states)
        self.assertEqual(feature.shape, (3, 3))
        self.assertEqual(gram.shape, (4, 4))
        self.assertEqual(token_cov.shape, (4, 4))
        np.testing.assert_allclose(feature, feature.T)
        np.testing.assert_allclose(gram, gram.T)
        np.testing.assert_allclose(token_cov, token_cov.T)

    def test_nonzero_spectra_match_feature_covariance(self):
        expected = np.linalg.eigvalsh(feature_covariance(self.states))
        expected = np.sort(expected[expected > 1e-12])[::-1]
        result = spectral_summary(self.states, top_k=3)
        np.testing.assert_allclose(
            result["top_eigenvalues"][: len(expected)], expected
        )

    def test_prefix_summaries_are_causal(self):
        original = prefix_summaries(self.states, [2, 3], top_k=2)
        changed = self.states.copy()
        changed[3] = 10_000
        after_future_change = prefix_summaries(changed, [2, 3], top_k=2)
        for key in ("rank", "trace", "effective_rank", "top_eigenvalues"):
            np.testing.assert_allclose(original[key], after_future_change[key])


if __name__ == "__main__":
    unittest.main()
