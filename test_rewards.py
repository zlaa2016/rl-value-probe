import unittest

from rewards import _parse_ifeval_constraint, ifeval_reward, ifeval_reward_details


class IFEvalConstraintParsingTest(unittest.TestCase):
    def test_dolci_nested_python_literal(self):
        row = {
            "ground_truth": [
                "[{'instruction_id': ['detectable_format:json_format'], "
                "'kwargs': [None]}]"
            ]
        }

        self.assertEqual(
            _parse_ifeval_constraint(row),
            {
                "instruction_id": ["detectable_format:json_format"],
                "kwargs": [None],
            },
        )

    def test_nested_json_serialization(self):
        row = {
            "ground_truth": [
                '[{"instruction_id": ["detectable_format:json_format"], '
                '"kwargs": [null]}]'
            ]
        }

        self.assertEqual(
            _parse_ifeval_constraint(row)["kwargs"],
            [None],
        )

    def test_exact_reward_accepts_real_dataset_shape(self):
        row = {
            "ground_truth": [
                "[{'instruction_id': ['detectable_format:json_format'], "
                "'kwargs': [None]}]"
            ]
        }

        self.assertEqual(ifeval_reward(row, '{"answer": 11}', None), 1.0)
        details = ifeval_reward_details(row, '{"answer": 11}', None)
        self.assertEqual(details["reward"], 1.0)
        self.assertEqual(
            details["constraint_results"],
            [{
                "instruction_id": "detectable_format:json_format",
                "kwargs": {},
                "passed": True,
            }],
        )


if __name__ == "__main__":
    unittest.main()
