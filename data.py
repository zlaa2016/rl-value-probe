from itertools import islice
from datasets import load_dataset

from config import DATASET, DATASET_REVISION, IF_ORIGINAL_DATASET


def load_if_prompts(n_prompts: int):
    """Stream n instruction-following RLVR prompts."""
    if n_prompts < 1:
        raise ValueError("n_prompts must be at least 1")

    dataset = load_dataset(
        DATASET,
        revision=DATASET_REVISION,
        split="train",
        streaming=True,
    )

    rows = (
        row for row in dataset
        if row.get("original_dataset") == IF_ORIGINAL_DATASET
    )
    result = list(islice(rows, n_prompts))

    if not result:
        raise RuntimeError(
            "No instruction-following prompts found for "
            f"original_dataset={IF_ORIGINAL_DATASET!r} at "
            f"dataset revision {DATASET_REVISION}."
        )

    return result
