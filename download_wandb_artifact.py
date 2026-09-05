import argparse
import json
from pathlib import Path

import numpy as np
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        help="Fully qualified W&B artifact, including :version or :latest.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    artifact = wandb.Api().artifact(args.artifact)
    output_dir = Path(artifact.download(root=args.output_dir))
    rollout_path = output_dir / "rollouts.jsonl"
    activation_path = output_dir / "activations.npz"

    if not rollout_path.exists() or not activation_path.exists():
        raise RuntimeError(
            "Artifact must contain rollouts.jsonl and activations.npz at its root."
        )

    with open(rollout_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with np.load(activation_path, allow_pickle=True) as activations:
        activation_ids = set(activations["rollout_ids"].astype(str))
        record_ids = {row["rollout_id"] for row in rows}
        if activation_ids != record_ids:
            raise RuntimeError(
                "Downloaded rollout and activation IDs do not match."
            )
        shape = activations["vectors"].shape
        stages = sorted(set(activations["model_stages"].astype(str)))

    print(f"Downloaded {artifact.name} -> {output_dir}")
    print(f"Rollouts: {len(rows)}")
    print(f"Activation vectors: {shape}")
    print(f"Model stages: {stages}")


if __name__ == "__main__":
    main()
