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
            "fractions": list(DEFAULT_FRACTIONS),
        },
    )

    rollout_records = []
    activation_vectors = []
    activation_rollout_ids = []
    activation_model_stages = []
    activation_layers = []
    activation_fractions = []

    for stage in args.models:
        model_name = MODELS[stage]
        print(f"\nLoading {stage}: {model_name}")
        model, tokenizer = load_model(model_name)

        for prompt_idx, row in enumerate(rows):
            prompt_ids = prompt_ids_from_row(row, tokenizer)

            for rollout_idx in range(args.n_rollouts):
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
                    "prompt_id": str(row.get("custom_id", prompt_idx)),
                    "rollout_index": rollout_idx,
                    "reward": reward,
                    "generated_text": result["text"],
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

                if run is not None:
                    run.log({
                        "rollout/reward": reward,
                        "rollout/generated_tokens": len(generated_ids),
                        "rollout/model_stage": stage,
                        "rollout/prompt_index": prompt_idx,
                        "rollout/sample_index": rollout_idx,
                    })

                for (layer, frac), vec in vectors.items():
                    activation_vectors.append(vec)
                    activation_rollout_ids.append(rollout_id)
                    activation_model_stages.append(stage)
                    activation_layers.append(layer)
                    activation_fractions.append(frac)

                print(
                    f"{stage:4s} prompt={prompt_idx:03d} "
                    f"rollout={rollout_idx} reward={reward:.2f} "
                    f"tokens={len(generated_ids)}"
                )

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(rollout_path, "w") as f:
        for record in rollout_records:
            f.write(json.dumps(record) + "\n")

    np.savez_compressed(
        activation_path,
        vectors=np.stack(activation_vectors).astype(np.float32),
        rollout_ids=np.array(activation_rollout_ids),
        model_stages=np.array(activation_model_stages),
        layers=np.array(activation_layers),
        fractions=np.array(activation_fractions),
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
