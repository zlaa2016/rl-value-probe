import argparse
from pathlib import Path

from run_mvp import load_outputs, save_outputs


def merge_output_directories(input_dirs, output_dir):
    """Merge independently generated stages into one probe-ready checkpoint."""
    rollout_records = []
    activation_vectors = []
    activation_rollout_ids = []
    activation_model_stages = []
    activation_layers = []
    activation_fractions = []

    for input_dir in map(Path, input_dirs):
        loaded = load_outputs(
            input_dir / "rollouts.jsonl",
            input_dir / "activations.npz",
        )
        if not loaded[0]:
            raise RuntimeError(f"No saved rollouts found in {input_dir}")

        rollout_records.extend(loaded[0])
        activation_vectors.extend(loaded[1])
        activation_rollout_ids.extend(loaded[2])
        activation_model_stages.extend(loaded[3])
        activation_layers.extend(loaded[4])
        activation_fractions.extend(loaded[5])

    rollout_ids = [record["rollout_id"] for record in rollout_records]
    if len(set(rollout_ids)) != len(rollout_ids):
        raise RuntimeError("Cannot merge duplicate rollout IDs.")

    keys = [
        (
            record["model_stage"],
            str(record["prompt_id"]),
            int(record["rollout_index"]),
        )
        for record in rollout_records
    ]
    if len(set(keys)) != len(keys):
        raise RuntimeError(
            "Cannot merge duplicate stage/prompt/rollout-index combinations."
        )

    record_id_set = set(rollout_ids)
    missing_records = set(activation_rollout_ids) - record_id_set
    if missing_records:
        raise RuntimeError(
            "Activation checkpoint references rollout IDs with no JSONL record."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_outputs(
        output_dir / "rollouts.jsonl",
        output_dir / "activations.npz",
        rollout_records,
        activation_vectors,
        activation_rollout_ids,
        activation_model_stages,
        activation_layers,
        activation_fractions,
    )
    return len(rollout_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="Directories containing rollouts.jsonl and activations.npz.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    count = merge_output_directories(args.input_dirs, args.output_dir)
    print(f"Merged {count} rollouts -> {args.output_dir}")


if __name__ == "__main__":
    main()
