import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from analysis_utils import policy_signal_matrix, regression_metrics
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


def paired_activation_deltas(acts, stage, layer, start_fraction, end_fraction):
    """Return h_end - h_start paired by rollout ID for one stage and layer."""
    def vectors_by_rollout(fraction):
        mask = (
            (acts["model_stages"] == stage)
            & (acts["layers"] == layer)
            & np.isclose(acts["fractions"], fraction)
        )
        return {
            rollout_id: vector
            for rollout_id, vector in zip(
                acts["rollout_ids"][mask],
                acts["vectors"][mask],
            )
        }

    start_vectors = vectors_by_rollout(start_fraction)
    end_vectors = vectors_by_rollout(end_fraction)
    rollout_ids = sorted(set(start_vectors) & set(end_vectors))

    if not rollout_ids:
        width = acts["vectors"].shape[1]
        return np.empty((0, width), dtype=acts["vectors"].dtype), np.array([])

    deltas = np.stack([
        end_vectors[rollout_id] - start_vectors[rollout_id]
        for rollout_id in rollout_ids
    ])
    return deltas, np.asarray(rollout_ids)


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


def evaluate_ridge_with_label_shuffle(
    X_train,
    y_train,
    X_test,
    y_test,
    n_shuffles,
    rng,
    alpha=10.0,
):
    """Evaluate a Ridge probe and a train-label permutation null on held-out data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    metrics = regression_metrics(y_test, model.predict(X_test_scaled))

    null_metrics = []
    for _ in range(n_shuffles):
        null_model = Ridge(alpha=alpha)
        null_model.fit(X_train_scaled, rng.permutation(y_train))
        null_metrics.append(
            regression_metrics(y_test, null_model.predict(X_test_scaled))
        )

    for metric_name in ("r2", "mse", "spearman_like"):
        values = np.asarray(
            [result[metric_name] for result in null_metrics],
            dtype=float,
        )
        finite = values[np.isfinite(values)]
        metrics[f"{metric_name}_null_mean"] = (
            float(finite.mean()) if len(finite) else float("nan")
        )
        metrics[f"{metric_name}_null_std"] = (
            float(finite.std()) if len(finite) else float("nan")
        )

    if n_shuffles:
        null_r2 = np.asarray([result["r2"] for result in null_metrics])
        null_mse = np.asarray([result["mse"] for result in null_metrics])
        metrics["r2_permutation_p"] = float(
            (1 + np.sum(null_r2 >= metrics["r2"])) / (n_shuffles + 1)
        )
        metrics["mse_permutation_p"] = float(
            (1 + np.sum(null_mse <= metrics["mse"])) / (n_shuffles + 1)
        )
    else:
        metrics["r2_permutation_p"] = float("nan")
        metrics["mse_permutation_p"] = float("nan")

    metrics["label_shuffles"] = int(n_shuffles)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default="outputs/rollouts.jsonl")
    parser.add_argument("--activations", default="outputs/activations.npz")
    parser.add_argument("--out", default="outputs/probe_results.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-label-shuffles",
        type=int,
        default=50,
        help="Number of train-label permutations used for the held-out null.",
    )
    add_wandb_args(parser)
    args = parser.parse_args()

    if args.n_label_shuffles < 0:
        parser.error("--n-label-shuffles must be non-negative")

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
            "seed": args.seed,
            "n_label_shuffles": args.n_label_shuffles,
        },
    )

    by_id = {r["rollout_id"]: r for r in rows}
    train_prompts, test_prompts = split_by_prompt(rows, seed=args.seed)
    shuffle_rng = np.random.default_rng(args.seed)

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

                metrics = evaluate_ridge_with_label_shuffle(
                    X_policy_train,
                    y_train,
                    X_policy_test,
                    y_test,
                    args.n_label_shuffles,
                    shuffle_rng,
                )
                results.append({
                    "model_stage": stage,
                    "start_fraction": np.nan,
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

                metrics = evaluate_ridge_with_label_shuffle(
                    X[train_mask],
                    y[train_mask],
                    X[test_mask],
                    y[test_mask],
                    args.n_label_shuffles,
                    shuffle_rng,
                )
                results.append({
                    "model_stage": stage,
                    "start_fraction": np.nan,
                    "fraction": frac,
                    "layer": layer,
                    "monitor": "activation_ridge",
                    **metrics,
                })

        # ----- within-rollout activation-update probes -----
        # The checkpoint is fixed; only trajectory time changes.
        for start_fraction, end_fraction in zip(fractions[:-1], fractions[1:]):
            for layer in layers:
                X_delta, ids = paired_activation_deltas(
                    acts,
                    stage,
                    layer,
                    start_fraction,
                    end_fraction,
                )
                if not len(ids):
                    continue

                y = np.array([by_id[i]["reward"] for i in ids], dtype=float)
                prompts = np.array([by_id[i]["prompt_id"] for i in ids])
                train_mask = np.array([p in train_prompts for p in prompts])
                test_mask = np.array([p in test_prompts for p in prompts])

                if train_mask.sum() < 4 or test_mask.sum() < 2:
                    continue

                metrics = evaluate_ridge_with_label_shuffle(
                    X_delta[train_mask],
                    y[train_mask],
                    X_delta[test_mask],
                    y[test_mask],
                    args.n_label_shuffles,
                    shuffle_rng,
                )
                results.append({
                    "model_stage": stage,
                    "start_fraction": start_fraction,
                    "fraction": end_fraction,
                    "layer": layer,
                    "monitor": "activation_delta_ridge",
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
            primary_layer = int(activation_df["layer"].max())
            primary = activation_df[
                activation_df["layer"] == primary_layer
            ].sort_values(["model_stage", "fraction"])
            labels = []
            values = []
            xs = []
            for stage, stage_df in primary.groupby("model_stage"):
                labels.append(stage)
                xs.append(stage_df["fraction"].tolist())
                values.append(stage_df["r2"].tolist())
            run.log({
                "probe/final_layer_activation_r2_over_trajectory": wandb.plot.line_series(
                    xs=xs,
                    ys=values,
                    keys=labels,
                    title=f"Layer {primary_layer} activation-probe R2 over trajectory",
                    xname="trajectory fraction",
                )
            })

        delta_df = df[df["monitor"] == "activation_delta_ridge"]
        if len(delta_df):
            primary_layer = int(delta_df["layer"].max())
            primary_delta = delta_df[
                delta_df["layer"] == primary_layer
            ].sort_values(["model_stage", "fraction"])
            labels = []
            values = []
            xs = []
            for stage, stage_df in primary_delta.groupby("model_stage"):
                labels.append(stage)
                xs.append(stage_df["fraction"].tolist())
                values.append(stage_df["r2"].tolist())
            run.log({
                "probe/final_layer_activation_delta_r2_over_trajectory": (
                    wandb.plot.line_series(
                        xs=xs,
                        ys=values,
                        keys=labels,
                        title=(
                            f"Layer {primary_layer} activation-update probe R2 "
                            "over trajectory"
                        ),
                        xname="end trajectory fraction",
                    )
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
