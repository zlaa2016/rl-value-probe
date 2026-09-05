import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "spearman_like": float(
            np.corrcoef(
                np.argsort(np.argsort(y_true)),
                np.argsort(np.argsort(y_pred)),
            )[0, 1]
        ) if len(y_true) > 2 else float("nan"),
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
