"""Token-wise residual-stream geometry for a future reward experiment.

Input is a single rollout at a single layer, represented by a matrix with one
row per token position and one column per residual-stream dimension.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _as_state_matrix(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states)
    if states.ndim != 2:
        raise ValueError(
            "states must have shape [token_positions, hidden_dimensions]"
        )
    if states.shape[0] < 2:
        raise ValueError("at least two token positions are required")
    if states.shape[1] < 1:
        raise ValueError("the hidden dimension must be non-empty")
    if not np.isfinite(states).all():
        raise ValueError("states contain NaN or infinite values")
    return states.astype(np.float64, copy=False)


def feature_covariance(states: np.ndarray) -> np.ndarray:
    """Return residual-dimension covariance across token positions, shape [D,D]."""
    states = _as_state_matrix(states)
    centered = states - states.mean(axis=0, keepdims=True)
    return centered.T @ centered / (len(states) - 1)


def token_gram(states: np.ndarray, center_positions: bool = True) -> np.ndarray:
    """Return scaled residual similarity between token positions, shape [T,T].

    With ``center_positions=True``, each residual dimension is centered across
    positions first. This is a centered Gram matrix, not transformer attention.
    """
    states = _as_state_matrix(states)
    if center_positions:
        states = states - states.mean(axis=0, keepdims=True)
    return states @ states.T / states.shape[1]


def token_covariance(states: np.ndarray) -> np.ndarray:
    """Return coordinate-wise covariance between position states, shape [T,T].

    Each token-position vector is centered across its residual coordinates.
    Residual dimensions are a learned basis rather than repeated observations,
    so this matrix should be treated as a geometric descriptor, not as an
    ordinary population covariance estimate.
    """
    states = _as_state_matrix(states)
    if states.shape[1] < 2:
        raise ValueError("token covariance requires at least two dimensions")
    row_centered = states - states.mean(axis=1, keepdims=True)
    return row_centered @ row_centered.T / (states.shape[1] - 1)


def spectral_summary(states: np.ndarray, top_k: int = 8) -> dict[str, np.ndarray | float | int]:
    """Summarize feature covariance through its nonzero spectrum.

    SVD obtains the nonzero covariance eigenvalues without constructing the
    potentially enormous [D,D] feature-covariance matrix.
    """
    states = _as_state_matrix(states)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    centered = states - states.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular_values**2 / (len(states) - 1)
    eigenvalues = eigenvalues[eigenvalues > np.finfo(float).eps]
    eigenvalues = np.sort(eigenvalues)[::-1]

    trace = float(eigenvalues.sum())
    squared_sum = float(eigenvalues @ eigenvalues)
    effective_rank = trace**2 / squared_sum if squared_sum else 0.0
    explained = eigenvalues / trace if trace else np.zeros_like(eigenvalues)

    padded_values = np.zeros(top_k, dtype=float)
    padded_explained = np.zeros(top_k, dtype=float)
    count = min(top_k, len(eigenvalues))
    padded_values[:count] = eigenvalues[:count]
    padded_explained[:count] = explained[:count]
    return {
        "token_count": int(states.shape[0]),
        "hidden_width": int(states.shape[1]),
        "rank": int(len(eigenvalues)),
        "trace": trace,
        "effective_rank": float(effective_rank),
        "top_eigenvalues": padded_values,
        "top_explained_ratio": padded_explained,
    }


def prefix_summaries(
    states: np.ndarray,
    prefix_lengths: list[int] | tuple[int, ...],
    top_k: int = 8,
) -> dict[str, np.ndarray]:
    """Return spectral summaries using only states observable at each prefix."""
    states = _as_state_matrix(states)
    lengths = np.asarray(prefix_lengths, dtype=int)
    if lengths.ndim != 1 or len(lengths) == 0:
        raise ValueError("prefix_lengths must be a non-empty one-dimensional list")
    if np.any(lengths < 2) or np.any(lengths > len(states)):
        raise ValueError("prefix lengths must lie between 2 and the token count")
    if len(set(lengths.tolist())) != len(lengths):
        raise ValueError("prefix lengths must be unique")

    summaries = [spectral_summary(states[:end], top_k=top_k) for end in lengths]
    return {
        "prefix_lengths": lengths,
        "rank": np.asarray([row["rank"] for row in summaries], dtype=int),
        "trace": np.asarray([row["trace"] for row in summaries], dtype=float),
        "effective_rank": np.asarray(
            [row["effective_rank"] for row in summaries], dtype=float
        ),
        "top_eigenvalues": np.stack(
            [row["top_eigenvalues"] for row in summaries]
        ),
        "top_explained_ratio": np.stack(
            [row["top_explained_ratio"] for row in summaries]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activations",
        required=True,
        help="NumPy .npy matrix with shape [token_positions, hidden_dimensions].",
    )
    parser.add_argument("--out", required=True, help="Destination .npz file.")
    parser.add_argument("--prefix-lengths", nargs="+", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--save-feature-covariance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    states = _as_state_matrix(np.load(args.activations, allow_pickle=False))
    prefixes = prefix_summaries(
        states,
        prefix_lengths=args.prefix_lengths,
        top_k=args.top_k,
    )
    output = {
        "token_gram": token_gram(states).astype(np.float32),
        "token_covariance": token_covariance(states).astype(np.float32),
        **prefixes,
    }
    if args.save_feature_covariance:
        output["feature_covariance"] = feature_covariance(states).astype(np.float32)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    print(f"Saved token geometry for {states.shape} states -> {output_path}")


if __name__ == "__main__":
    main()
