import unittest

from model_specs import (
    parse_model_spec,
    resolve_model_specs,
    validate_resume_model_specs,
)


class ModelSpecTest(unittest.TestCase):
    def test_parses_intermediate_hugging_face_revision(self):
        self.assertEqual(
            parse_model_spec(
                "pretrain10k=allenai/Olmo-3-1025-7B@stage1-step10000"
            ),
            {
                "label": "pretrain10k",
                "model_name": "allenai/Olmo-3-1025-7B",
                "revision": "stage1-step10000",
            },
        )

    def test_omitted_revision_uses_main(self):
        self.assertEqual(
            parse_model_spec("sft=allenai/Olmo-3-7B-Think-SFT")["revision"],
            "main",
        )

    def test_legacy_named_models_still_work(self):
        specs = resolve_model_specs(["base", "rlvr"], None)
        self.assertEqual([spec["label"] for spec in specs], ["base", "rlvr"])
        self.assertTrue(all(spec["revision"] == "main" for spec in specs))

    def test_labels_must_be_unique(self):
        with self.assertRaisesRegex(ValueError, "unique label"):
            resolve_model_specs(
                None,
                [
                    "same=allenai/Olmo-3-1025-7B@stage1-step10000",
                    "same=allenai/Olmo-3-1025-7B@stage2-step10000",
                ],
            )

    def test_resume_rejects_reused_label_with_different_revision(self):
        records = [{
            "model_stage": "base_early",
            "model_name": "allenai/Olmo-3-1025-7B",
            "model_revision": "stage1-step10000",
        }]
        specs = [{
            "label": "base_early",
            "model_name": "allenai/Olmo-3-1025-7B",
            "revision": "stage2-step10000",
        }]
        with self.assertRaisesRegex(ValueError, "different|Cannot resume"):
            validate_resume_model_specs(records, specs)


if __name__ == "__main__":
    unittest.main()
