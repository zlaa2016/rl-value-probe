import argparse
import os

import numpy as np

from config import WANDB_ENTITY, WANDB_PROJECT


def add_wandb_args(parser: argparse.ArgumentParser):
    """Add shared W&B options without ever accepting an API key as an argument."""
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
        help="Use online logging, local-only offline logging, or no W&B logging.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", WANDB_ENTITY),
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
    )
    parser.add_argument("--wandb-run-name", default=None)


def init_wandb(args, job_type, config=None):
    """Initialize W&B using `wandb login` or WANDB_API_KEY credentials."""
    if args.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging is enabled but wandb is not installed. "
            "Run `pip install wandb`, or pass `--wandb-mode disabled`."
        ) from exc

    try:
        return wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            job_type=job_type,
            mode=args.wandb_mode,
            config=config or {},
        )
    except Exception as exc:
        if args.wandb_mode == "online":
            raise RuntimeError(
                "Could not start online W&B logging. Run `wandb login --verify`, "
                "set WANDB_API_KEY, or use `--wandb-mode offline`."
            ) from exc
        raise


def log_rollout_trajectories(run, records, fractions):
    """Log saved or freshly generated rollout trajectories to a W&B run."""
    if run is None or not records:
        return

    import wandb

    fractions = sorted(float(fraction) for fraction in fractions)
    signal_names = (
        "mean_token_logprob",
        "geomean_token_prob",
        "sequence_logprob",
        "mean_entropy",
        "mean_top1_prob",
        "last_token_logprob",
        "last_token_entropy",
        "last_top1_prob",
    )
    columns = [
        "rollout_id",
        "model_stage",
        "model_name",
        "prompt_id",
        "rollout_index",
        "fraction",
        "reward",
        "generated_tokens",
        "partial_text",
        *signal_names,
    ]
    table_data = []
    for record in records:
        generated_tokens = record["token_positions"]["1.00"] + 1
        for fraction in fractions:
            key = f"{fraction:.2f}"
            signals = record["policy_signals"][key]
            table_data.append([
                record["rollout_id"],
                record["model_stage"],
                record["model_name"],
                record["prompt_id"],
                record["rollout_index"],
                fraction,
                record["reward"],
                generated_tokens,
                record["partial_text"][key],
                *[signals[name] for name in signal_names],
            ])

    run.log({
        "trajectory/table": wandb.Table(columns=columns, data=table_data),
    })

    stages = sorted({record["model_stage"] for record in records})
    for signal_name in ("mean_token_logprob", "mean_entropy", "mean_top1_prob"):
        series = []
        labels = []
        for stage in stages:
            values = []
            for fraction in fractions:
                key = f"{fraction:.2f}"
                stage_values = [
                    record["policy_signals"][key][signal_name]
                    for record in records
                    if record["model_stage"] == stage
                ]
                values.append(float(np.mean(stage_values)))
            series.append(values)
            labels.append(stage)
        run.log({
            f"trajectory/{signal_name}": wandb.plot.line_series(
                xs=fractions,
                ys=series,
                keys=labels,
                title=f"{signal_name} over rollout trajectory",
                xname="trajectory fraction",
            )
        })
