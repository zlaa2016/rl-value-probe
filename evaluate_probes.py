import argparse
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from baselines import policy_signal_matrix, regression_metrics
from probe import load_activations, load_rollouts
from tracking import add_wandb_args, init_wandb


INTERNAL_PROBES = (
    "current_ridge",
    "current_rbf",
    "temporal_history_ridge",
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


def fit_predict(X_train, y_train, X_test, model_kind, alpha):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_kind == "ridge":
        model = Ridge(alpha=alpha)
    elif model_kind == "rbf":
        distances = pairwise_distances(X_train, metric="sqeuclidean")
        positive = distances[np.triu_indices_from(distances, k=1)]
        positive = positive[positive > 0]
        median_distance = float(np.median(positive)) if len(positive) else 1.0
        gamma = 1.0 / max(median_distance, 1e-12)
        model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=float)


def leave_one_prompt_out_predictions(
    X,
    y,
    prompts,
    model_kind,
    alpha,
    rng=None,
    shuffle_train_labels=False,
):
    """Predict every rollout from a model that never trained on its prompt."""
    unique_prompts = sorted(set(prompts))
    if len(unique_prompts) < 3:
        raise ValueError("At least three distinct prompts are required.")

    predictions = np.full(len(y), np.nan, dtype=float)
    for prompt in unique_prompts:
        test_mask = prompts == prompt
        train_mask = ~test_mask
        y_train = y[train_mask]
        if shuffle_train_labels:
            if rng is None:
                raise ValueError("An RNG is required for shuffled-label controls.")
            y_train = rng.permutation(y_train)
        predictions[test_mask] = fit_predict(
            X[train_mask],
            y_train,
            X[test_mask],
            model_kind=model_kind,
            alpha=alpha,
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError("Some leave-one-prompt-out predictions are missing.")
    return predictions


def permutation_null(X, y, prompts, n_shuffles, alpha, rng):
    null_r2 = []
    for _ in range(n_shuffles):
        prediction = leave_one_prompt_out_predictions(
            X,
            y,
            prompts,
            model_kind="ridge",
            alpha=alpha,
            rng=rng,
            shuffle_train_labels=True,
        )
        null_r2.append(regression_metrics(y, prediction)["r2"])
    return np.asarray(null_r2, dtype=float)


def append_evaluation(
    metric_rows,
    prediction_rows,
    X,
    y,
    prompts,
    rollout_ids,
    stage,
    layer,
    fraction,
    probe_name,
    model_kind,
    alpha,
    n_label_shuffles,
    rng,
):
    prediction = leave_one_prompt_out_predictions(
        X,
        y,
        prompts,
        model_kind=model_kind,
        alpha=alpha,
    )
    metrics = regression_metrics(y, prediction)

    if n_label_shuffles and probe_name == "current_ridge":
        null_r2 = permutation_null(
            X,
            y,
            prompts,
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
        "n_rollouts": len(y),
        "n_prompts": len(set(prompts)),
        "label_shuffles": n_label_shuffles if probe_name == "current_ridge" else 0,
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
            rollout_ids,
            prompts,
            y,
            prediction,
        )
    ])


def plot_instruction_trajectories(predictions, output_dir):
    """Save one final-layer out-of-prompt trajectory plot per instruction."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--rbf-alpha", type=float, default=1.0)
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
                stage,
                -1,
                fraction,
                "policy_confidence_ridge",
                "ridge",
                args.ridge_alpha,
                0,
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
                    stage,
                    layer,
                    fraction,
                    "current_ridge",
                    "ridge",
                    args.ridge_alpha,
                    args.n_label_shuffles,
                    rng,
                )
                append_evaluation(
                    metric_rows,
                    prediction_rows,
                    X,
                    y,
                    prompts,
                    rollout_ids,
                    stage,
                    layer,
                    fraction,
                    "current_rbf",
                    "rbf",
                    args.rbf_alpha,
                    0,
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
                    stage,
                    layer,
                    fraction,
                    "temporal_history_ridge",
                    "ridge",
                    args.ridge_alpha,
                    0,
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
                        stage,
                        layer,
                        fraction,
                        "activation_delta_ridge",
                        "ridge",
                        args.ridge_alpha,
                        0,
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
    predictions_path = output_dir / "oof_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    image_paths = plot_instruction_trajectories(
        predictions,
        output_dir / "instruction_trajectories",
    )

    run = init_wandb(
        args,
        job_type="probe-evaluation",
        config={
            "validation": "leave-one-prompt-out",
            "ridge_alpha": args.ridge_alpha,
            "rbf_alpha": args.rbf_alpha,
            "n_label_shuffles": args.n_label_shuffles,
            "seed": args.seed,
        },
    )
    if run is not None:
        import wandb

        run.log({
            "evaluation/metrics": wandb.Table(dataframe=metrics),
            "evaluation/oof_predictions": wandb.Table(dataframe=predictions),
            "evaluation/instruction_trajectories": wandb.Table(
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
                "evaluation/final_layer_oof_r2": wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=labels,
                    title=f"Final-layer out-of-prompt R2 (layer {final_layer})",
                    xname="trajectory fraction",
                )
            })
        run.summary["rollouts"] = len(rows)
        run.summary["prompts"] = len({str(row["prompt_id"]) for row in rows})
        run.summary["model_stages"] = stages
        artifact = wandb.Artifact(
            f"{args.wandb_run_name or 'probe-evaluation'}-outputs",
            type="probe-results",
        )
        artifact.add_dir(str(output_dir))
        run.log_artifact(artifact)
        run.finish()

    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved out-of-prompt predictions -> {predictions_path}")
    print(f"Saved {len(image_paths)} instruction trajectory plots")


if __name__ == "__main__":
    main()
