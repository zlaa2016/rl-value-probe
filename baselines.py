import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def _average_ranks(values):
    """Return zero-based average ranks, including correct handling of ties."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    target_variance = float(np.var(y_true))
    prediction_variance = float(np.var(y_pred))
    r2 = (
        float(r2_score(y_true, y_pred))
        if not np.allclose(y_true, y_true[0])
        else float("nan")
    )
    spearman = (
        float(np.corrcoef(_average_ranks(y_true), _average_ranks(y_pred))[0, 1])
        if len(y_true) > 2
        and not np.allclose(y_true, y_true[0])
        and not np.allclose(y_pred, y_pred[0])
        else float("nan")
    )
    return {
        "r2": r2,
        "mse": float(mean_squared_error(y_true, y_pred)),
        "spearman_like": spearman,
        "target_variance": target_variance,
        "prediction_variance": prediction_variance,
    }


POLICY_SIGNAL_KEYS = (
    "mean_token_logprob",
    "mean_entropy",
    "mean_top1_prob",
    "last_token_logprob",
    "last_token_entropy",
    "last_top1_prob",
)


def policy_signal_matrix(rows, fraction_key):
    """
    Observable model-confidence features for a rollout prefix.

    These are not trained rewards. They are raw policy statistics derived
    from the model's next-token distribution.
    """
    return np.asarray([
        [
            row["policy_signals"][fraction_key][key]
            for key in POLICY_SIGNAL_KEYS
        ]
        for row in rows
    ], dtype=float)
