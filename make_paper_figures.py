import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_probes import (
    plot_constraint_pass_rates,
)
from audit_evidence import trajectory_association_audit


STAGES = ("base", "sft", "dpo", "rlvr")
STAGE_LABELS = ("Base", "SFT", "DPO", "RLVR")


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def executive_summary_figure(rollouts, output_path):
    frame = pd.DataFrame(rollouts)
    stage_means = frame.groupby("model_stage")["reward"].mean().reindex(STAGES)
    prompt_means = (
        frame.pivot_table(
            index="prompt_id",
            columns="model_stage",
            values="reward",
            aggfunc="mean",
        )
        .reindex(columns=STAGES)
        .sort_index()
    )
    prompt_labels = [value.split("-request-")[-1] for value in prompt_means.index]

    figure = plt.figure(figsize=(11, 5.8), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(0.8, 1.35),
        height_ratios=(5.0, 0.75),
    )
    reward_axis = figure.add_subplot(grid[0, 0])
    prompt_axis = figure.add_subplot(grid[0, 1])
    note_axis = figure.add_subplot(grid[1, :])

    bars = reward_axis.bar(
        STAGE_LABELS,
        stage_means.to_numpy(),
        color=plt.get_cmap("Blues")(np.linspace(0.45, 0.85, len(STAGES))),
    )
    reward_axis.set_ylim(0, 0.16)
    reward_axis.set_ylabel("Mean terminal reward")
    reward_axis.set_xlabel("Checkpoint")
    reward_axis.set_title("A. Aggregate reward")
    reward_axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, stage_means):
        reward_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    image = prompt_axis.imshow(
        prompt_means.to_numpy(),
        vmin=0,
        vmax=0.4,
        cmap="Blues",
        aspect="auto",
    )
    prompt_axis.set_xticks(range(len(STAGES)), STAGE_LABELS)
    prompt_axis.set_yticks(range(len(prompt_labels)), prompt_labels)
    prompt_axis.set_xlabel("Checkpoint")
    prompt_axis.set_ylabel("Prompt ID")
    prompt_axis.set_title("B. Reward is prompt-specific")
    for row in range(len(prompt_means)):
        for column in range(len(STAGES)):
            value = prompt_means.iloc[row, column]
            prompt_axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.25 else "black",
            )
    colorbar = figure.colorbar(image, ax=prompt_axis, shrink=0.82)
    colorbar.set_label("Mean terminal reward")

    note_axis.axis("off")
    note_axis.text(
        0.5,
        0.58,
        "Prompt identity explains 80–100% of reward variance"
        "    •    Only 4/24 prompt × checkpoint cells vary across rollouts"
        "    •    Held-out R² is undefined",
        ha="center",
        va="center",
        fontweight="bold",
    )
    figure.suptitle("Key evidential findings from the completed experiment", fontsize=16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def trajectory_association_figure(trajectories, output_path):
    audit = trajectory_association_audit(trajectories)
    feature_rows = (
        ("confidence", "Generated-token confidence"),
        ("activation_change_rms", "Activation change (RMS)"),
    )
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(13, 7),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    for row_index, (feature, feature_label) in enumerate(feature_rows):
        for column_index, (stage, stage_label) in enumerate(
            zip(STAGES, STAGE_LABELS)
        ):
            axis = axes[row_index, column_index]
            selected = audit[
                (audit["feature"] == feature)
                & (audit["model_stage"] == stage)
            ].sort_values("fraction")
            axis.plot(
                selected["fraction"],
                selected["pooled_pearson"],
                marker="o",
                linewidth=2,
                label="Pooled",
            )
            axis.plot(
                selected["fraction"],
                selected["within_prompt_pearson"],
                marker="s",
                linestyle="--",
                linewidth=2,
                label="Within-prompt",
            )
            axis.axhline(0, color="0.5", linewidth=1)
            axis.grid(alpha=0.18)
            axis.set_ylim(-0.65, 0.8)
            axis.set_xticks([0.25, 0.50, 0.75, 1.00])
            if row_index == 0:
                axis.set_title(stage_label)
            if column_index == 0:
                axis.set_ylabel(f"{feature_label}\nPearson correlation")
            if row_index == 1:
                axis.set_xlabel("Trajectory fraction")
            if selected["within_prompt_pearson"].isna().all():
                axis.text(
                    0.625,
                    -0.52,
                    "Within-prompt undefined",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="0.35",
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Pooled associations change after controlling for prompt identity",
        y=0.985,
        fontsize=15,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.09,
        top=0.82,
        wspace=0.06,
        hspace=0.10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def example_prompt_figure(trajectories, prompt_id, output_path):
    selected = trajectories[trajectories["prompt_id"] == prompt_id]
    if selected.empty:
        raise ValueError(f"Prompt {prompt_id!r} is not present in the trajectory table.")

    measures = (
        ("reward", "Terminal reward"),
        ("confidence", "Generated-token confidence"),
        ("activation_change_rms", "Activation change (RMS)"),
    )
    colors = dict(zip(STAGES, plt.get_cmap("tab10").colors[:4]))
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9, 8),
        sharex=True,
        constrained_layout=False,
    )
    for axis, (measure, label) in zip(axes, measures):
        for stage in STAGES:
            stage_rows = selected[selected["model_stage"] == stage]
            summary = stage_rows.groupby("fraction")[measure].agg(["mean", "std"])
            x = summary.index.to_numpy()
            mean = summary["mean"].to_numpy()
            std = summary["std"].fillna(0).to_numpy()
            axis.plot(
                x,
                mean,
                marker="o",
                linewidth=2,
                color=colors[stage],
                label=stage.upper(),
            )
            axis.fill_between(
                x,
                mean - std,
                mean + std,
                color=colors[stage],
                alpha=0.12,
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Trajectory fraction")
    axes[-1].set_xticks([0.25, 0.50, 0.75, 1.00])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(
        "Example prompt 639-88: digestion with three constraints",
        y=0.98,
        fontsize=15,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
    )
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.08, top=0.88, hspace=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouts",
        default="outputs/all-models-six-5ro/rollouts-annotated.jsonl",
    )
    parser.add_argument(
        "--analysis-dir",
        default="outputs/analysis/all-models-six-5ro",
    )
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument(
        "--example-prompt-id",
        default="IF_multi_constraints_upto5_filtered-request-639-88",
    )
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    rollouts = load_jsonl(args.rollouts)
    trajectories = pd.read_csv(analysis_dir / "instruction_trajectories.csv")
    constraint_difficulty = pd.read_csv(
        analysis_dir / "constraint_difficulty.csv"
    )

    executive_summary_figure(rollouts, output_dir / "executive_summary.png")
    trajectory_association_figure(
        trajectories,
        output_dir / "trajectory_correlation_with_terminal_reward.png",
    )
    plot_constraint_pass_rates(
        constraint_difficulty,
        output_dir / "constraint_pass_rates.png",
    )
    example_prompt_figure(
        trajectories,
        args.example_prompt_id,
        output_dir / "example_prompt_639_88.png",
    )
    for path in sorted(output_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
