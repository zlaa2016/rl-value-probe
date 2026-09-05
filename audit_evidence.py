"""Audit whether apparent trajectory signals support the paper's hypotheses.

This analysis deliberately does not fit another high-dimensional probe. It asks
whether checkpoint effects are broad across prompts, whether reward varies
within prompt/checkpoint cells, whether pooled trajectory correlations survive
prompt fixed effects, and how much terminal reward is already visible to the
external verifier from partial text.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rewards import ifeval_reward_details


STAGES = ("base", "sft", "dpo", "rlvr")


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_correlation(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def checkpoint_effect_audit(rollouts, n_bootstrap=200_000, seed=0):
    frame = pd.DataFrame(rollouts)
    prompt_means = frame.pivot_table(
        index="prompt_id",
        columns="model_stage",
        values="reward",
        aggfunc="mean",
    ).reindex(columns=STAGES)
    rng = np.random.default_rng(seed)
    records = []
    for stage in STAGES[1:]:
        differences = (prompt_means[stage] - prompt_means["base"]).to_numpy()
        bootstrap_means = np.mean(
            rng.choice(
                differences,
                size=(n_bootstrap, len(differences)),
                replace=True,
            ),
            axis=1,
        )
        sign_flipped_means = [
            np.mean(differences * np.asarray(signs))
            for signs in itertools.product((-1, 1), repeat=len(differences))
        ]
        observed = float(np.mean(differences))
        permutation_p = np.mean(
            np.abs(sign_flipped_means) >= abs(observed) - 1e-12
        )
        records.append({
            "comparison": f"{stage}_minus_base",
            "mean_prompt_difference": observed,
            "median_prompt_difference": float(np.median(differences)),
            "positive_prompts": int(np.sum(differences > 0)),
            "negative_prompts": int(np.sum(differences < 0)),
            "unchanged_prompts": int(np.sum(differences == 0)),
            "prompt_bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
            "prompt_bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
            "paired_sign_flip_p": float(permutation_p),
        })
    return pd.DataFrame(records)


def reward_variation_audit(rollouts):
    frame = pd.DataFrame(rollouts)
    records = []
    for stage, stage_rows in frame.groupby("model_stage"):
        prompt_mean = stage_rows.groupby("prompt_id")["reward"].transform("mean")
        total_ss = float(np.sum((stage_rows["reward"] - stage_rows["reward"].mean()) ** 2))
        between_ss = float(np.sum((prompt_mean - stage_rows["reward"].mean()) ** 2))
        within_ss = float(np.sum((stage_rows["reward"] - prompt_mean) ** 2))
        varying_cells = sum(
            group["reward"].nunique() > 1
            for _, group in stage_rows.groupby("prompt_id")
        )
        records.append({
            "model_stage": stage,
            "prompt_cells": int(stage_rows["prompt_id"].nunique()),
            "varying_prompt_cells": int(varying_cells),
            "between_prompt_variance_share": between_ss / total_ss if total_ss else np.nan,
            "within_prompt_variance_share": within_ss / total_ss if total_ss else np.nan,
        })
    return pd.DataFrame(records).sort_values("model_stage")


def trajectory_association_audit(trajectories):
    records = []
    for (stage, fraction), group in trajectories.groupby(
        ["model_stage", "fraction"]
    ):
        for feature in ("confidence", "activation_change_rms"):
            prompt_centered_feature = group[feature] - group.groupby("prompt_id")[
                feature
            ].transform("mean")
            prompt_centered_reward = group["reward"] - group.groupby("prompt_id")[
                "reward"
            ].transform("mean")
            leave_one_prompt_out = []
            for prompt_id in group["prompt_id"].unique():
                subset = group[group["prompt_id"] != prompt_id]
                value = safe_correlation(subset[feature], subset["reward"])
                if np.isfinite(value):
                    leave_one_prompt_out.append(value)
            records.append({
                "model_stage": stage,
                "fraction": float(fraction),
                "feature": feature,
                "pooled_pearson": safe_correlation(group[feature], group["reward"]),
                "within_prompt_pearson": safe_correlation(
                    prompt_centered_feature,
                    prompt_centered_reward,
                ),
                "leave_one_prompt_out_min": (
                    min(leave_one_prompt_out) if leave_one_prompt_out else np.nan
                ),
                "leave_one_prompt_out_max": (
                    max(leave_one_prompt_out) if leave_one_prompt_out else np.nan
                ),
            })
    return pd.DataFrame(records).sort_values(
        ["feature", "model_stage", "fraction"]
    )


def prefix_verifier_audit(rollouts):
    prefix_records = []
    for rollout in rollouts:
        ground_truth = {
            "instruction_id": [
                item["instruction_id"] for item in rollout["constraint_results"]
            ],
            "kwargs": [item["kwargs"] for item in rollout["constraint_results"]],
        }
        for fraction, partial_text in rollout["partial_text"].items():
            prefix_reward = ifeval_reward_details(
                row={"ground_truth": ground_truth},
                generated_text=partial_text,
                generated_ids=[],
            )["reward"]
            prefix_records.append({
                "rollout_id": rollout["rollout_id"],
                "model_stage": rollout["model_stage"],
                "prompt_id": rollout["prompt_id"],
                "fraction": float(fraction),
                "prefix_verifier_reward": prefix_reward,
                "terminal_reward": rollout["reward"],
            })

    prefixes = pd.DataFrame(prefix_records)
    summaries = []
    for (stage, fraction), group in prefixes.groupby(
        ["model_stage", "fraction"]
    ):
        centered_prefix = group["prefix_verifier_reward"] - group.groupby(
            "prompt_id"
        )["prefix_verifier_reward"].transform("mean")
        centered_terminal = group["terminal_reward"] - group.groupby(
            "prompt_id"
        )["terminal_reward"].transform("mean")
        summaries.append({
            "model_stage": stage,
            "fraction": float(fraction),
            "mean_prefix_verifier_reward": float(
                group["prefix_verifier_reward"].mean()
            ),
            "pooled_terminal_correlation": safe_correlation(
                group["prefix_verifier_reward"], group["terminal_reward"]
            ),
            "within_prompt_terminal_correlation": safe_correlation(
                centered_prefix, centered_terminal
            ),
            "exact_terminal_match_rate": float(
                np.isclose(
                    group["prefix_verifier_reward"], group["terminal_reward"]
                ).mean()
            ),
        })
    return prefixes, pd.DataFrame(summaries).sort_values(
        ["model_stage", "fraction"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouts",
        default="outputs/all-models-six-5ro/rollouts-annotated.jsonl",
    )
    parser.add_argument(
        "--trajectories",
        default="outputs/analysis/all-models-six-5ro/instruction_trajectories.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/all-models-six-5ro/evidence-audit",
    )
    parser.add_argument("--n-bootstrap", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts = load_jsonl(args.rollouts)
    trajectories = pd.read_csv(args.trajectories)

    outputs = {
        "checkpoint_effect_audit.csv": checkpoint_effect_audit(
            rollouts,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        ),
        "reward_variation_audit.csv": reward_variation_audit(rollouts),
        "trajectory_association_audit.csv": trajectory_association_audit(
            trajectories
        ),
    }
    prefix_rows, prefix_summary = prefix_verifier_audit(rollouts)
    outputs["prefix_verifier_rows.csv"] = prefix_rows
    outputs["prefix_verifier_audit.csv"] = prefix_summary

    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        print(f"Saved {len(frame)} rows -> {path}")


if __name__ == "__main__":
    main()
