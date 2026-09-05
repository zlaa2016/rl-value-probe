import argparse
import gc
import json
from pathlib import Path
import uuid

import numpy as np
import torch

from activations import extract_states_and_signals
from config import (
    DEFAULT_FRACTIONS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MODELS,
)
from data import load_if_prompts
from generate import generate_rollout, load_model, prompt_ids_from_row
from rewards import ifeval_reward
from tracking import add_wandb_args, init_wandb, log_rollout_trajectories


def partial_texts(tokenizer, generated_ids, fractions):
    parts = {}
    n = len(generated_ids)

    for frac in fractions:
        end = max(1, min(n, round(frac * n)))
        text = tokenizer.decode(
            generated_ids[:end],
            skip_special_tokens=True,
        )
        parts[f"{frac:.2f}"] = text

    return parts


def save_outputs(
    rollout_path,
    activation_path,
    rollout_records,
    activation_vectors,
    activation_rollout_ids,
    activation_model_stages,
    activation_layers,
    activation_fractions,
):
    """Atomically checkpoint every completed rollout and its activations."""
    if not rollout_records or not activation_vectors:
        return

    rollout_tmp = rollout_path.with_name(f".{rollout_path.name}.tmp")
    with open(rollout_tmp, "w") as f:
        for record in rollout_records:
            f.write(json.dumps(record) + "\n")
    rollout_tmp.replace(rollout_path)

    activation_tmp = activation_path.with_name(f".{activation_path.name}.tmp")
    with open(activation_tmp, "wb") as f:
        np.savez_compressed(
            f,
            vectors=np.stack(activation_vectors).astype(np.float32),
            rollout_ids=np.array(activation_rollout_ids),
            model_stages=np.array(activation_model_stages),
            layers=np.array(activation_layers),
            fractions=np.array(activation_fractions),
        )
    activation_tmp.replace(activation_path)


def load_outputs(rollout_path, activation_path):
    """Load a matching rollout/activation checkpoint for --resume."""
    rollouts_exist = rollout_path.exists()
    activations_exist = activation_path.exists()
    if rollouts_exist != activations_exist:
        raise RuntimeError(
            "Cannot resume because only one checkpoint file exists: "
            f"{rollout_path} / {activation_path}"
        )

    if not rollouts_exist:
        return ([], [], [], [], [], [])

    with open(rollout_path) as f:
        rollout_records = [json.loads(line) for line in f if line.strip()]

    with np.load(activation_path, allow_pickle=True) as data:
        activation_vectors = list(data["vectors"].copy())
        activation_rollout_ids = data["rollout_ids"].astype(str).tolist()
        activation_model_stages = data["model_stages"].astype(str).tolist()
        activation_layers = data["layers"].astype(int).tolist()
        activation_fractions = data["fractions"].astype(float).tolist()

    activation_lengths = {
        len(activation_vectors),
        len(activation_rollout_ids),
        len(activation_model_stages),
        len(activation_layers),
        len(activation_fractions),
    }
    if len(activation_lengths) != 1:
        raise RuntimeError("Activation checkpoint arrays have mismatched lengths.")

    return (
        rollout_records,
        activation_vectors,
        activation_rollout_ids,
        activation_model_stages,
        activation_layers,
        activation_fractions,
    )


def rollout_key(stage, row, prompt_idx, rollout_idx):
    prompt_id = str(row.get("custom_id", prompt_idx))
    return stage, prompt_id, int(rollout_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["base", "sft"],
        choices=list(MODELS),
    )
    parser.add_argument("--n-prompts", type=int, default=5)
    parser.add_argument("--n-rollouts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a checkpoint in --output-dir. --n-rollouts is the target "
            "total per prompt; completed stage/prompt/index tuples are skipped."
        ),
    )
    add_wandb_args(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_path = out_dir / "rollouts.jsonl"
    activation_path = out_dir / "activations.npz"

    rows = load_if_prompts(args.n_prompts)
    print(f"Loaded {len(rows)} IF RLVR prompts.")

    run = init_wandb(
        args,
        job_type="rollout-generation",
        config={
            "models": args.models,
            "model_checkpoints": {stage: MODELS[stage] for stage in args.models},
            "n_prompts": args.n_prompts,
            "n_rollouts": args.n_rollouts,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "resume": args.resume,
            "fractions": list(DEFAULT_FRACTIONS),
        },
    )

    if args.resume:
        (
            rollout_records,
            activation_vectors,
            activation_rollout_ids,
            activation_model_stages,
            activation_layers,
            activation_fractions,
        ) = load_outputs(rollout_path, activation_path)
        print(f"Resuming from {len(rollout_records)} completed rollouts.")
    else:
        rollout_records = []
        activation_vectors = []
        activation_rollout_ids = []
        activation_model_stages = []
        activation_layers = []
        activation_fractions = []

    existing_keys = {
        (
            record["model_stage"],
            str(record["prompt_id"]),
            int(record["rollout_index"]),
        )
        for record in rollout_records
    }
    if len(existing_keys) != len(rollout_records):
        raise RuntimeError(
            "The rollout checkpoint contains duplicate stage/prompt/index rows."
        )

    for stage in args.models:
        pending = any(
            rollout_key(stage, row, prompt_idx, rollout_idx) not in existing_keys
            for prompt_idx, row in enumerate(rows)
            for rollout_idx in range(args.n_rollouts)
        )
        if not pending:
            print(
                f"\nSkipping {stage}: already has {args.n_rollouts} "
                "rollouts per requested prompt."
            )
            continue

        model_name = MODELS[stage]
        print(f"\nLoading {stage}: {model_name}")
        model, tokenizer = load_model(model_name)

        for prompt_idx, row in enumerate(rows):
            prompt_ids = prompt_ids_from_row(row, tokenizer)

            for rollout_idx in range(args.n_rollouts):
                key = rollout_key(stage, row, prompt_idx, rollout_idx)
                if key in existing_keys:
                    continue

                if run is not None:
                    run.log({
                        "progress/rollouts_started": len(rollout_records) + 1,
                        "progress/model_stage": stage,
                        "progress/phase": "generation",
                    })

                result = generate_rollout(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                generated_ids = result["generated_ids"]
                if len(generated_ids) == 0:
                    print(f"Skipping empty rollout: {stage=} {prompt_idx=} {rollout_idx=}")
                    continue

                reward = ifeval_reward(
                    row=row,
                    generated_text=result["text"],
                    generated_ids=generated_ids,
                )

                if run is not None:
                    # Log the terminal reward before the more memory-intensive
                    # activation replay so W&B shows experiment progress early.
                    run.log({
                        "rollout/reward": reward,
                        "rollout/generated_tokens": len(generated_ids),
                        "progress/model_stage": stage,
                        "progress/phase": "activation_extraction",
                    })

                vectors, token_positions, policy_signals = extract_states_and_signals(
                    model=model,
                    prompt_ids=prompt_ids,
                    generated_ids=generated_ids,
                    fractions=DEFAULT_FRACTIONS,
                )

                rollout_id = str(uuid.uuid4())

                record = {
                    "rollout_id": rollout_id,
                    "model_stage": stage,
                    "model_name": model_name,
                    "prompt_id": key[1],
                    "rollout_index": rollout_idx,
                    "reward": reward,
                    "generated_text": result["text"],
                    # Preserve the exact trajectory for future teacher-forced
                    # replay through every checkpoint.
                    "generated_token_ids": generated_ids.tolist(),
                    "partial_text": partial_texts(
                        tokenizer,
                        generated_ids,
                        DEFAULT_FRACTIONS,
                    ),
                    "policy_signals": policy_signals,
                    "token_positions": {
                        f"{k:.2f}": int(v)
                        for k, v in token_positions.items()
                    },
                }
                rollout_records.append(record)
                existing_keys.add(key)

                for (layer, frac), vec in vectors.items():
                    activation_vectors.append(vec)
                    activation_rollout_ids.append(rollout_id)
                    activation_model_stages.append(stage)
                    activation_layers.append(layer)
                    activation_fractions.append(frac)

                save_outputs(
                    rollout_path,
                    activation_path,
                    rollout_records,
                    activation_vectors,
                    activation_rollout_ids,
                    activation_model_stages,
                    activation_layers,
                    activation_fractions,
                )

                if run is not None:
                    run.log({
                        "progress/rollouts_completed": len(rollout_records),
                        "progress/model_stage": stage,
                        "progress/phase": "rollout_complete",
                        "rollout/model_stage": stage,
                        "rollout/prompt_index": prompt_idx,
                        "rollout/sample_index": rollout_idx,
                    })

                print(
                    f"{stage:4s} prompt={prompt_idx:03d} "
                    f"rollout={rollout_idx} reward={reward:.2f} "
                    f"tokens={len(generated_ids)}"
                )

        if run is not None and rollout_records:
            # Persist each finished model stage outside the Kaggle session.
            run.save(
                str(rollout_path.resolve()),
                base_path=str(out_dir.resolve()),
                policy="now",
            )
            run.save(
                str(activation_path.resolve()),
                base_path=str(out_dir.resolve()),
                policy="now",
            )

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_outputs(
        rollout_path,
        activation_path,
        rollout_records,
        activation_vectors,
        activation_rollout_ids,
        activation_model_stages,
        activation_layers,
        activation_fractions,
    )

    if run is not None:
        log_rollout_trajectories(run, rollout_records, DEFAULT_FRACTIONS)

        run.summary["rollouts_completed"] = len(rollout_records)
        run.summary["rollouts_path"] = str(rollout_path)
        run.summary["activations_path"] = str(activation_path)
        run.finish()

    print(f"\nSaved {len(rollout_records)} rollouts -> {rollout_path}")
    print(f"Saved activations -> {activation_path}")


if __name__ == "__main__":
    main()
