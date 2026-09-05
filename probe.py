import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import policy_signal_matrix, regression_metrics
from tracking import add_wandb_args, init_wandb, log_rollout_trajectories


def load_rollouts(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_activations(path):
    data = np.load(path, allow_pickle=True)
    return {
        "vectors": data["vectors"],
        "rollout_ids": data["rollout_ids"].astype(str),
        "model_stages": data["model_stages"].astype(str),
        "layers": data["layers"].astype(int),
        "fractions": data["fractions"].astype(float),
    }


def split_by_prompt(rows, test_size=0.25, seed=0):
    """
    One split shared across models/fractions.
    Different sampled rollouts of one prompt never cross train/test.
    """
    prompt_ids = np.array([r["prompt_id"] for r in rows])
    groups = prompt_ids

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )

    dummy = np.zeros(len(rows))
    train_idx, test_idx = next(splitter.split(dummy, groups=groups))

    train_prompts = set(prompt_ids[train_idx])
    test_prompts = set(prompt_ids[test_idx])
    return train_prompts, test_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default="outputs/rollouts.jsonl")
    parser.add_argument("--activations", default="outputs/activations.npz")
    parser.add_argument("--out", default="outputs/probe_results.csv")
    add_wandb_args(parser)
    args = parser.parse_args()

    rows = load_rollouts(args.rollouts)
    acts = load_activations(args.activations)

    run = init_wandb(
        args,
        job_type="probe-evaluation",
        config={
            "rollouts": args.rollouts,
            "activations": args.activations,
            "split": "grouped-by-prompt",
            "test_size": 0.25,
            "ridge_alpha": 10.0,
        },
    )

    by_id = {r["rollout_id"]: r for r in rows}
    train_prompts, test_prompts = split_by_prompt(rows)

    results = []

    stages = sorted(set(acts["model_stages"]))
    layers = sorted(set(acts["layers"]))
    fractions = sorted(set(acts["fractions"]))

    for stage in stages:
        for frac in fractions:
            stage_rows = [
                r for r in rows
                if r["model_stage"] == stage
            ]

            train_rows = [
                r for r in stage_rows
                if r["prompt_id"] in train_prompts
            ]
            test_rows = [
                r for r in stage_rows
                if r["prompt_id"] in test_prompts
            ]

            key = f"{frac:.2f}"
            y_train = np.array([r["reward"] for r in train_rows], dtype=float)
            y_test = np.array([r["reward"] for r in test_rows], dtype=float)

            if len(train_rows) >= 4 and len(test_rows) >= 2:
                # ----- policy-confidence diagnostic -----
                # Train/calibrate only on train prompts, evaluate on held-out prompts.
                X_policy_train = policy_signal_matrix(train_rows, key)
                X_policy_test = policy_signal_matrix(test_rows, key)

                policy_model = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=10.0),
                )
                policy_model.fit(X_policy_train, y_train)
                pred = policy_model.predict(X_policy_test)
                metrics = regression_metrics(y_test, pred)
                results.append({
                    "model_stage": stage,
                    "fraction": frac,
                    "layer": -1,
                    "monitor": "policy_confidence",
                    **metrics,
                })

            # ----- activation probes -----
            for layer in layers:
                mask = (
                    (acts["model_stages"] == stage)
                    & (acts["layers"] == layer)
                    & np.isclose(acts["fractions"], frac)
                )

                X = acts["vectors"][mask]
                ids = acts["rollout_ids"][mask]

                y = np.array([by_id[i]["reward"] for i in ids], dtype=float)
                prompts = np.array([by_id[i]["prompt_id"] for i in ids])

                train_mask = np.array([p in train_prompts for p in prompts])
                test_mask = np.array([p in test_prompts for p in prompts])

                if train_mask.sum() < 4 or test_mask.sum() < 2:
                    continue

                probe = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=10.0),
                )
                probe.fit(X[train_mask], y[train_mask])
                pred = probe.predict(X[test_mask])

                metrics = regression_metrics(y[test_mask], pred)
                results.append({
                    "model_stage": stage,
                    "fraction": frac,
                    "layer": layer,
                    "monitor": "activation_ridge",
                    **metrics,
                })

    df = pd.DataFrame(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    if run is not None and len(df):
        import wandb

        clean_df = df.replace({np.nan: None})
        run.log({"probe/results": wandb.Table(dataframe=clean_df)})

        activation_df = df[df["monitor"] == "activation_ridge"]
        if len(activation_df):
            best = (
                activation_df.sort_values("r2", ascending=False)
                .groupby(["model_stage", "fraction"], as_index=False)
                .first()
                .sort_values(["model_stage", "fraction"])
            )
            labels = []
            values = []
            xs = []
            for stage, stage_df in best.groupby("model_stage"):
                labels.append(stage)
                xs.append(stage_df["fraction"].tolist())
                values.append(stage_df["r2"].tolist())
            run.log({
                "probe/best_activation_r2_over_trajectory": wandb.plot.line_series(
                    xs=xs,
                    ys=values,
                    keys=labels,
                    title="Best activation-probe R2 over trajectory",
                    xname="trajectory fraction",
                )
            })

        run.summary["result_rows"] = len(df)
        run.summary["results_path"] = args.out

    if run is not None:
        fractions = sorted({
            float(key)
            for row in rows
            for key in row["policy_signals"]
        })
        log_rollout_trajectories(run, rows, fractions)
        run.finish()

    if len(df):
        print(df.sort_values(["model_stage", "fraction", "monitor", "layer"]).to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
