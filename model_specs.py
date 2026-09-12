"""Parse model repositories and optional Hugging Face revisions."""

from __future__ import annotations

import re

from config import MODELS


_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_model_spec(value: str) -> dict[str, str]:
    """Parse ``LABEL=MODEL_ID[@REVISION]`` into a serializable record."""
    if "=" not in value:
        raise ValueError(
            f"Invalid model spec {value!r}; expected LABEL=MODEL_ID[@REVISION]."
        )
    label, location = value.split("=", 1)
    if not _LABEL_PATTERN.fullmatch(label):
        raise ValueError(
            f"Invalid model label {label!r}; use letters, numbers, '.', '_', or '-'."
        )

    if "@" in location:
        model_name, revision = location.rsplit("@", 1)
    else:
        model_name, revision = location, "main"
    if not model_name or not revision:
        raise ValueError(
            f"Invalid model spec {value!r}; model ID and revision must be non-empty."
        )
    return {
        "label": label,
        "model_name": model_name,
        "revision": revision,
    }


def resolve_model_specs(
    named_models: list[str] | None,
    custom_specs: list[str] | None,
) -> list[dict[str, str]]:
    """Resolve legacy stage names or arbitrary revision-aware specifications."""
    if named_models and custom_specs:
        raise ValueError("Use either named models or custom model specs, not both.")

    if custom_specs:
        specs = [parse_model_spec(value) for value in custom_specs]
    else:
        stages = named_models or ["base", "sft"]
        specs = [
            {
                "label": stage,
                "model_name": MODELS[stage],
                "revision": "main",
            }
            for stage in stages
        ]

    labels = [spec["label"] for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("Every model specification must have a unique label.")
    return specs


def validate_resume_model_specs(
    rollout_records: list[dict],
    specs: list[dict[str, str]],
) -> None:
    """Prevent a resumed label from silently referring to different weights."""
    expected = {spec["label"]: spec for spec in specs}
    for record in rollout_records:
        label = record["model_stage"]
        if label not in expected:
            continue
        spec = expected[label]
        recorded_revision = record.get("model_revision", "main")
        if (
            record.get("model_name") != spec["model_name"]
            or recorded_revision != spec["revision"]
        ):
            raise ValueError(
                f"Cannot resume label {label!r}: saved rollouts use "
                f"{record.get('model_name')}@{recorded_revision}, but the requested "
                f"model is {spec['model_name']}@{spec['revision']}."
            )
