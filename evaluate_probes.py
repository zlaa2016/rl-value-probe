import argparse
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from baselines import policy_signal_matrix, regression_metrics
from probe import load_activations, load_rollouts, split_by_prompt
from tracking import add_wandb_args, init_wandb


INTERNAL_PROBES = (
    "current_ridge",
    "temporal_history_ridge",
    "current_plus_confidence_ridge",
)


def activation_lookup(activations):
    return {
        (stage, int(layer), float(fraction), rollout_id): vector
        for vector, rollout_id, stage, layer, fraction in zip(
            activations["vectors"],
            activations["rollout_ids"],
            activations["model_stages"],
            activations["layers"],
            activations["fractions"],
        )
    }


def activation_features(rows, lookup, stage, layer, fractions, fraction, mode):
    """Build current-state, cumulative-history, or adjacent-delta features."""
    selected_rows = [row for row in rows if row["model_stage"] == stage]
    history = [value for value in fractions if value <= fraction]
    features = []

    for row in selected_rows:
        rollout_id = row["rollout_id"]
        if mode == "current":
            feature = lookup[(stage, layer, fraction, rollout_id)]
        elif mode == "history":
            feature = np.concatenate([
                lookup[(stage, layer, value, rollout_id)]
                for value in history
            ])
        elif mode == "delta":
            position = fractions.index(fraction)
            if position == 0:
                raise ValueError("A delta feature requires a previous fraction.")
            start = fractions[position - 1]
            feature = (
                lookup[(stage, layer, fraction, rollout_id)]
                - lookup[(stage, layer, start, rollout_id)]
            )
        else:
            raise ValueError(f"Unknown activation feature mode: {mode}")
        features.append(feature)

    return (
        np.asarray(features),
        np.asarray([row["reward"] for row in selected_rows], dtype=float),
        np.asarray([str(row["prompt_id"]) for row in selected_rows]),
        np.asarray([row["rollout_id"] for row in selected_rows]),
    )


def fit_predict(X_train, y_train, X_test, alpha):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=float)


def heldout_predictions(
    X,
    y,
    prompts,
    train_prompts,
    test_prompts,
    alpha,
    rng=None,
    shuffle_train_labels=False,
):
    """Fit once on train prompts and predict only the fixed held-out prompts."""
    train_mask = np.asarray([prompt in train_prompts for prompt in prompts])
    test_mask = np.asarray([prompt in test_prompts for prompt in prompts])
    if train_mask.sum() < 4 or test_mask.sum() < 2:
        raise ValueError("The fixed prompt split has too few train/test rollouts.")

    y_train = y[train_mask]
    if shuffle_train_labels:
        if rng is None:
            raise ValueError("An RNG is required for shuffled-label controls.")
        y_train = rng.permutation(y_train)
    prediction = fit_predict(
        X[train_mask],
        y_train,
        X[test_mask],
        alpha=alpha,
    )
    return prediction, train_mask, test_mask


def permutation_null(
    X,
    y,
    prompts,
    train_prompts,
    test_prompts,
    n_shuffles,
    alpha,
    rng,
):
    null_r2 = []
    for _ in range(n_shuffles):
        prediction, _, test_mask = heldout_predictions(
            X,
            y,
            prompts,
            train_prompts,
            test_prompts,
            alpha=alpha,
            rng=rng,
            shuffle_train_labels=True,
        )
        null_r2.append(regression_metrics(y[test_mask], prediction)["r2"])
    return np.asarray(null_r2, dtype=float)


def append_evaluation(
    metric_rows,
    prediction_rows,
    X,
    y,
    prompts,
    rollout_ids,
    train_prompts,
    test_prompts,
    stage,
    layer,
    fraction,
    probe_name,
    alpha,
    n_label_shuffles,
    rng,
):
    prediction, train_mask, test_mask = heldout_predictions(
        X,
        y,
        prompts,
        train_prompts,
        test_prompts,
        alpha=alpha,
    )
    metrics = regression_metrics(y[test_mask], prediction)

    if n_label_shuffles:
        null_r2 = permutation_null(
            X,
            y,
            prompts,
            train_prompts,
            test_prompts,
            n_label_shuffles,
            alpha,
            rng,
        )
        finite = null_r2[np.isfinite(null_r2)]
        metrics["r2_null_mean"] = float(finite.mean()) if len(finite) else math.nan
        metrics["r2_null_std"] = float(finite.std()) if len(finite) else math.nan
        metrics["r2_permutation_p"] = (
            float((1 + np.sum(finite >= metrics["r2"])) / (len(finite) + 1))
            if len(finite) and np.isfinite(metrics["r2"])
            else math.nan
        )
    else:
        metrics["r2_null_mean"] = math.nan
        metrics["r2_null_std"] = math.nan
        metrics["r2_permutation_p"] = math.nan

    metric_rows.append({
        "model_stage": stage,
        "layer": layer,
        "fraction": fraction,
        "probe": probe_name,
        "n_train_rollouts": int(train_mask.sum()),
        "n_test_rollouts": int(test_mask.sum()),
        "n_train_prompts": len(train_prompts),
        "n_test_prompts": len(test_prompts),
        "label_shuffles": n_label_shuffles,
        **metrics,
    })
    prediction_rows.extend([
        {
            "rollout_id": rollout_id,
            "prompt_id": prompt,
            "model_stage": stage,
            "layer": layer,
            "fraction": fraction,
            "probe": probe_name,
            "reward": target,
            "prediction": estimate,
        }
        for rollout_id, prompt, target, estimate in zip(
            rollout_ids[test_mask],
            prompts[test_mask],
            y[test_mask],
            prediction,
        )
    ])


def plot_instruction_trajectories(predictions, output_dir):
    """Save one final-layer held-out prediction plot per test instruction."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Instruction trajectory plots require matplotlib. "
            "Run `pip install -r requirements.txt`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    final_layer = int(predictions["layer"].max())
    selected = predictions[
        (predictions["layer"] == final_layer)
        & predictions["probe"].isin(INTERNAL_PROBES)
    ]
    paths = []

    for index, (prompt_id, prompt_df) in enumerate(selected.groupby("prompt_id")):
        instruction_text = str(prompt_df["instruction_text"].iloc[0]).strip()
        title_text = instruction_text[:140] if instruction_text else str(prompt_id)
        fig, axes = plt.subplots(
            len(INTERNAL_PROBES),
            1,
            figsize=(8, 8),
            sharex=True,
            constrained_layout=True,
        )
        for axis, probe_name in zip(axes, INTERNAL_PROBES):
            probe_df = prompt_df[prompt_df["probe"] == probe_name]
            for stage, stage_df in probe_df.groupby("model_stage"):
                summary = (
                    stage_df.groupby("fraction")
                    .agg(
                        prediction=("prediction", "mean"),
                        prediction_std=("prediction", "std"),
                        reward=("reward", "mean"),
                    )
                    .reset_index()
                )
                x = summary["fraction"].to_numpy()
                mean = summary["prediction"].to_numpy()
                std = summary["prediction_std"].fillna(0).to_numpy()
                line = axis.plot(x, mean, marker="o", label=stage)[0]
                axis.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    color=line.get_color(),
                    alpha=0.12,
                )
                axis.scatter(
                    [1.0],
                    [summary["reward"].iloc[-1]],
                    color=line.get_color(),
                    marker="x",
                    s=55,
                )
            axis.axhline(0.0, color="0.75", linewidth=0.8)
            axis.axhline(1.0, color="0.75", linewidth=0.8)
            axis.set_ylabel("Predicted reward")
            axis.set_title(probe_name.replace("_", " "))
        axes[0].legend(title="checkpoint", ncol=4, fontsize=8)
        axes[-1].set_xlabel("Trajectory fraction")
        fig.suptitle(f"{title_text} · final layer {final_layer}")

        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(prompt_id))[:60]
        path = output_dir / f"instruction-{index:03d}-{safe_id}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append((str(prompt_id), path))
    return paths


def descriptive_trajectory_frame(rows, lookup, final_layer, fractions):
    """Describe reward, confidence, and hidden-state movement for every prompt."""
    records = []
    for row in rows:
        stage = row["model_stage"]
        rollout_id = row["rollout_id"]
        start = lookup[(stage, final_layer, fractions[0], rollout_id)]
        width = max(len(start), 1)
        for fraction in fractions:
            current = lookup[(stage, final_layer, fraction, rollout_id)]
            records.append({
                "rollout_id": rollout_id,
                "prompt_id": str(row["prompt_id"]),
                "instruction_text": str(row.get("prompt_text", "")),
                "model_stage": stage,
                "fraction": fraction,
                "reward": float(row["reward"]),
                "confidence": float(
                    row["policy_signals"][f"{fraction:.2f}"][
                        "geomean_token_prob"
                    ]
                ),
                "activation_change_rms": float(
                    np.linalg.norm(current - start) / math.sqrt(width)
                ),
            })
    return pd.DataFrame(records)


def faithfulness_correlations(descriptive):
    rows = []
    pairs = (
        ("confidence", "reward"),
        ("activation_change_rms", "reward"),
        ("confidence", "activation_change_rms"),
    )
    for (stage, fraction), group in descriptive.groupby(
        ["model_stage", "fraction"]
    ):
        for first, second in pairs:
            rows.append({
                "model_stage": stage,
                "fraction": fraction,
                "variable_1": first,
                "variable_2": second,
                "n": len(group),
                "pearson": group[first].corr(group[second], method="pearson"),
                "spearman": group[first].corr(group[second], method="spearman"),
            })
    return pd.DataFrame(rows)


def constraint_analysis(rows, descriptive):
    constraint_rows = []
    for row in rows:
        for result in row.get("constraint_results", []):
            constraint_rows.append({
                "rollout_id": row["rollout_id"],
                "prompt_id": str(row["prompt_id"]),
                "model_stage": row["model_stage"],
                "instruction_id": result["instruction_id"],
                "passed": float(result["passed"]),
            })
    if not constraint_rows:
        return pd.DataFrame(), pd.DataFrame()

    constraints = pd.DataFrame(constraint_rows)
    difficulty = (
        constraints.groupby(["model_stage", "instruction_id"])
        .agg(
            examples=("passed", "size"),
            prompts=("prompt_id", "nunique"),
            pass_rate=("passed", "mean"),
        )
        .reset_index()
    )
    merged = descriptive.merge(
        constraints,
        on=["rollout_id", "prompt_id", "model_stage"],
        how="inner",
    )
    correlation_rows = []
    for (stage, fraction, instruction_id), group in merged.groupby(
        ["model_stage", "fraction", "instruction_id"]
    ):
        for variable in ("confidence", "activation_change_rms"):
            correlation_rows.append({
                "model_stage": stage,
                "fraction": fraction,
                "instruction_id": instruction_id,
                "variable": variable,
                "n": len(group),
                "prompts": group["prompt_id"].nunique(),
                "pearson_with_pass": group[variable].corr(
                    group["passed"], method="pearson"
                ),
                "spearman_with_pass": group[variable].corr(
                    group["passed"], method="spearman"
                ),
            })
    return difficulty, pd.DataFrame(correlation_rows)


def plot_descriptive_instruction_trajectories(descriptive, output_dir):
    """Plot every instruction, without using fitted probe predictions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Instruction trajectory plots require matplotlib. "
            "Run `pip install -r requirements.txt`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    measures = (
        ("reward", "Terminal reward"),
        ("confidence", "Generated-token confidence"),
        ("activation_change_rms", "Activation change (RMS)"),
    )
    paths = []
    for index, (prompt_id, prompt_df) in enumerate(descriptive.groupby("prompt_id")):
        instruction_text = str(prompt_df["instruction_text"].iloc[0]).strip()
        title_text = instruction_text[:140] if instruction_text else str(prompt_id)
        fig, axes = plt.subplots(
            len(measures),
            1,
            figsize=(8, 8),
            sharex=True,
            constrained_layout=True,
        )
        for axis, (measure, label) in zip(axes, measures):
            for stage, stage_df in prompt_df.groupby("model_stage"):
                summary = (
                    stage_df.groupby("fraction")[measure]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                x = summary["fraction"].to_numpy()
                mean = summary["mean"].to_numpy()
                std = summary["std"].fillna(0).to_numpy()
                line = axis.plot(x, mean, marker="o", label=stage)[0]
                axis.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    color=line.get_color(),
                    alpha=0.12,
                )
            axis.set_ylabel(label)
        axes[0].legend(title="checkpoint", ncol=4, fontsize=8)
        axes[-1].set_xlabel("Trajectory fraction")
        fig.suptitle(title_text)

        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(prompt_id))[:60]
        path = output_dir / f"instruction-{index:03d}-{safe_id}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append((str(prompt_id), path))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--n-label-shuffles", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    add_wandb_args(parser)
    args = parser.parse_args()

    if args.n_label_shuffles < 0:
        parser.error("--n-label-shuffles must be non-negative")

    rows = load_rollouts(args.rollouts)
    activations = load_activations(args.activations)
    lookup = activation_lookup(activations)
    stages = sorted(set(activations["model_stages"]))
    layers = sorted(set(activations["layers"]))
    fractions = sorted(set(activations["fractions"]))
    train_prompts, test_prompts = split_by_prompt(
        rows,
        test_size=args.test_size,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    metric_rows = []
    prediction_rows = []

    for stage in stages:
        stage_rows = [row for row in rows if row["model_stage"] == stage]
        for fraction in fractions:
            fraction_key = f"{fraction:.2f}"
            policy_X = policy_signal_matrix(stage_rows, fraction_key)
            policy_y = np.asarray([row["reward"] for row in stage_rows], dtype=float)
            policy_prompts = np.asarray([str(row["prompt_id"]) for row in stage_rows])
            policy_ids = np.asarray([row["rollout_id"] for row in stage_rows])
            append_evaluation(
                metric_rows,
                prediction_rows,
                policy_X,
                policy_y,
                policy_prompts,
                policy_ids,
                train_prompts,
                test_prompts,
                stage,
                -1,
                fraction,
                "policy_confidence_ridge",
                args.ridge_alpha,
                args.n_label_shuffles,
                rng,
            )

            for layer in layers:
                X, y, prompts, rollout_ids = activation_features(
                    rows, lookup, stage, layer, fractions, fraction, "current"
                )
                append_evaluation(
                    metric_rows,
                    prediction_rows,
                    X,
                    y,
                    prompts,
                    rollout_ids,
                    train_prompts,
                    test_prompts,
                    stage,
                    layer,
                    fraction,
                    "current_ridge",
                    args.ridge_alpha,
                    args.n_label_shuffles,
                    rng,
                )

                history_X, _, _, _ = activation_features(
                    rows, lookup, stage, layer, fractions, fraction, "history"
                )
                append_evaluation(
                    metric_rows,
                    prediction_rows,
                    history_X,
                    y,
                    prompts,
                    rollout_ids,
                    train_prompts,
                    test_prompts,
                    stage,
                    layer,
                    fraction,
                    "temporal_history_ridge",
                    args.ridge_alpha,
                    args.n_label_shuffles,
                    rng,
                )

                combined_X = np.concatenate([X, policy_X], axis=1)
                append_evaluation(
                    metric_rows,
                    prediction_rows,
                    combined_X,
                    y,
                    prompts,
                    rollout_ids,
                    train_prompts,
                    test_prompts,
                    stage,
                    layer,
                    fraction,
                    "current_plus_confidence_ridge",
                    args.ridge_alpha,
                    args.n_label_shuffles,
                    rng,
                )

                if fraction != fractions[0]:
                    delta_X, _, _, _ = activation_features(
                        rows, lookup, stage, layer, fractions, fraction, "delta"
                    )
                    append_evaluation(
                        metric_rows,
                        prediction_rows,
                        delta_X,
                        y,
                        prompts,
                        rollout_ids,
                        train_prompts,
                        test_prompts,
                        stage,
                        layer,
                        fraction,
                        "activation_delta_ridge",
                        args.ridge_alpha,
                        args.n_label_shuffles,
                        rng,
                    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    instruction_text = {
        str(row["prompt_id"]): str(row.get("prompt_text", ""))
        for row in rows
    }
    predictions["instruction_text"] = predictions["prompt_id"].map(
        instruction_text
    ).fillna("")
    metrics_path = output_dir / "probe_metrics.csv"
    predictions_path = output_dir / "heldout_predictions.csv"
    descriptive = descriptive_trajectory_frame(
        rows,
        lookup,
        int(max(layers)),
        fractions,
    )
    descriptive_path = output_dir / "instruction_trajectories.csv"
    correlations = faithfulness_correlations(descriptive)
    correlations_path = output_dir / "faithfulness_correlations.csv"
    constraint_difficulty, constraint_correlations = constraint_analysis(
        rows,
        descriptive,
    )
    constraint_difficulty_path = output_dir / "constraint_difficulty.csv"
    constraint_correlations_path = output_dir / "constraint_correlations.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    descriptive.to_csv(descriptive_path, index=False)
    correlations.to_csv(correlations_path, index=False)
    if len(constraint_difficulty):
        constraint_difficulty.to_csv(constraint_difficulty_path, index=False)
        constraint_correlations.to_csv(constraint_correlations_path, index=False)
    image_paths = plot_instruction_trajectories(
        predictions,
        output_dir / "heldout_probe_trajectories",
    )
    descriptive_image_paths = plot_descriptive_instruction_trajectories(
        descriptive,
        output_dir / "instruction_trajectories",
    )

    run = init_wandb(
        args,
        job_type="probe-evaluation",
        config={
            "validation": "single-fixed-prompt-holdout",
            "test_size": args.test_size,
            "ridge_alpha": args.ridge_alpha,
            "n_label_shuffles": args.n_label_shuffles,
            "seed": args.seed,
        },
    )
    if run is not None:
        import wandb

        run.log({
            "evaluation/metrics": wandb.Table(dataframe=metrics),
            "evaluation/heldout_predictions": wandb.Table(dataframe=predictions),
            "evaluation/faithfulness_correlations": wandb.Table(
                dataframe=correlations
            ),
            "evaluation/heldout_probe_trajectories": wandb.Table(
                columns=["prompt_id", "instruction_text", "trajectory"],
                data=[
                    [
                        prompt_id,
                        instruction_text.get(prompt_id, ""),
                        wandb.Image(str(path)),
                    ]
                    for prompt_id, path in image_paths
                ],
            ),
            "evaluation/instruction_trajectories": wandb.Table(
                columns=["prompt_id", "instruction_text", "trajectory"],
                data=[
                    [
                        prompt_id,
                        instruction_text.get(prompt_id, ""),
                        wandb.Image(str(path)),
                    ]
                    for prompt_id, path in descriptive_image_paths
                ],
            ),
        })
        if len(constraint_difficulty):
            run.log({
                "evaluation/constraint_difficulty": wandb.Table(
                    dataframe=constraint_difficulty
                ),
                "evaluation/constraint_correlations": wandb.Table(
                    dataframe=constraint_correlations
                ),
            })
        final_layer = int(metrics["layer"].max())
        curve_data = metrics[
            (metrics["layer"] == final_layer)
            & metrics["probe"].isin(
                [*INTERNAL_PROBES, "activation_delta_ridge"]
            )
        ].sort_values(["model_stage", "probe", "fraction"])
        labels = []
        xs = []
        ys = []
        for (stage, probe_name), group in curve_data.groupby(
            ["model_stage", "probe"]
        ):
            labels.append(f"{stage} · {probe_name}")
            xs.append(group["fraction"].tolist())
            ys.append(group["r2"].tolist())
        if labels:
            run.log({
                "evaluation/final_layer_heldout_r2": wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=labels,
                    title=f"Final-layer held-out R2 (layer {final_layer})",
                    xname="trajectory fraction",
                )
            })
        run.summary["rollouts"] = len(rows)
        run.summary["prompts"] = len({str(row["prompt_id"]) for row in rows})
        run.summary["model_stages"] = stages
        run.summary["train_prompts"] = sorted(train_prompts)
        run.summary["test_prompts"] = sorted(test_prompts)
        artifact = wandb.Artifact(
            f"{args.wandb_run_name or 'probe-evaluation'}-outputs",
            type="probe-results",
        )
        artifact.add_dir(str(output_dir))
        run.log_artifact(artifact)
        run.finish()

    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved held-out predictions -> {predictions_path}")
    print(f"Saved correlations -> {correlations_path}")
    if len(constraint_difficulty):
        print(f"Saved constraint difficulty -> {constraint_difficulty_path}")
        print(f"Saved constraint correlations -> {constraint_correlations_path}")
    print(f"Saved {len(descriptive_image_paths)} instruction trajectory plots")
    print(f"Saved {len(image_paths)} held-out probe trajectory plots")


if __name__ == "__main__":
    main()
