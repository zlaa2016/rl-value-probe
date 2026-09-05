import unittest

import pandas as pd

from audit_evidence import reward_variation_audit, trajectory_association_audit


class EvidenceAuditTest(unittest.TestCase):
    def test_prompt_identity_can_explain_all_reward_variation(self):
        rollouts = [
            {"model_stage": "base", "prompt_id": "a", "reward": 0.0},
            {"model_stage": "base", "prompt_id": "a", "reward": 0.0},
            {"model_stage": "base", "prompt_id": "b", "reward": 1.0},
            {"model_stage": "base", "prompt_id": "b", "reward": 1.0},
        ]
        result = reward_variation_audit(rollouts).iloc[0]
        self.assertEqual(result["varying_prompt_cells"], 0)
        self.assertEqual(result["between_prompt_variance_share"], 1.0)
        self.assertEqual(result["within_prompt_variance_share"], 0.0)

    def test_prompt_centering_removes_between_prompt_association(self):
        trajectories = pd.DataFrame([
            {
                "model_stage": "base",
                "prompt_id": "a",
                "fraction": 1.0,
                "reward": 0.0,
                "confidence": 0.1,
                "activation_change_rms": 0.2,
            },
            {
                "model_stage": "base",
                "prompt_id": "a",
                "fraction": 1.0,
                "reward": 0.0,
                "confidence": 0.2,
                "activation_change_rms": 0.3,
            },
            {
                "model_stage": "base",
                "prompt_id": "b",
                "fraction": 1.0,
                "reward": 1.0,
                "confidence": 0.8,
                "activation_change_rms": 0.9,
            },
            {
                "model_stage": "base",
                "prompt_id": "b",
                "fraction": 1.0,
                "reward": 1.0,
                "confidence": 0.9,
                "activation_change_rms": 1.0,
            },
        ])
        result = trajectory_association_audit(trajectories)
        confidence = result[result["feature"] == "confidence"].iloc[0]
        self.assertGreater(confidence["pooled_pearson"], 0.9)
        self.assertTrue(pd.isna(confidence["within_prompt_pearson"]))


if __name__ == "__main__":
    unittest.main()
