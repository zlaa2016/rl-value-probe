"""A small, readable POMDP sanity check for trajectory-to-reward prediction.

This file is deliberately separate from the language-model experiment.  It
illustrates one theoretical idea: when the environment dynamics are known,
the Bayesian belief over a hidden state is a sufficient summary of the
observable history for predicting future outcomes.

The experiment compares three prefix representations:

1. the latest noisy observation alone;
2. the complete observation/action history so far; and
3. the exact Bayesian belief P(hidden state is good | history).

Each representation is given to the same logistic-regression probe, which
predicts the binary terminal reward.  The resulting AUCs are a conceptual
sanity check only.  They do *not* establish that transformer activations are
Bayesian beliefs or that probe accuracy demonstrates causal faithfulness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tracking import add_wandb_args, init_wandb


# The unobserved environment state is either bad (0) or good (1).
BAD, GOOD = 0, 1

# The agent can preserve the current state or try to flip it.
PRESERVE, FLIP = 0, 1

# TRANSITION[action, current_state, next_state].
#
# PRESERVE usually leaves the hidden state unchanged.  FLIP usually changes
# it.  Neither action is perfectly reliable, so uncertainty remains.
TRANSITION = np.array(
    [
        [[0.90, 0.10], [0.10, 0.90]],  # preserve
        [[0.10, 0.90], [0.90, 0.10]],  # flip
    ],
    dtype=float,
)

# OBSERVATION[hidden_state, observed_symbol].  The observation agrees with the
# hidden state 75% of the time and is misleading 25% of the time.
OBSERVATION = np.array(
    [
        [0.75, 0.25],  # hidden state is bad
        [0.25, 0.75],  # hidden state is good
    ],
    dtype=float,
)

# Before seeing anything, both hidden states are equally likely.
PRIOR = np.array([0.50, 0.50], dtype=float)


def normalize(probabilities: np.ndarray) -> np.ndarray:
    """Return a probability vector whose entries sum to one."""
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("Cannot normalize a zero-probability event.")
    return probabilities / total


def incorporate_observation(
    predicted_belief: np.ndarray, observation: int
) -> np.ndarray:
    """Apply Bayes' rule after receiving one noisy observation.

    posterior(s) is proportional to
        P(observation | state=s) * predicted_belief(s).
    """
    likelihood = OBSERVATION[:, observation]
    return normalize(predicted_belief * likelihood)


def predict_next_belief(belief: np.ndarray, action: int) -> np.ndarray:
    """Propagate the belief through the chosen state-transition matrix."""
    return belief @ TRANSITION[action]


def choose_action(belief: np.ndarray) -> int:
    """A simple fixed policy: preserve likely-good states; otherwise flip."""
    return PRESERVE if belief[GOOD] >= 0.5 else FLIP


def sample_categorical(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    """Sample an integer category from a probability vector."""
    return int(rng.choice(len(probabilities), p=probabilities))


def simulate_episode(horizon: int, rng: np.random.Generator) -> dict:
    """Simulate one complete trajectory and return its prefix information.

    The agent first observes the initial state.  Between observations it:
      1. chooses an action from its current belief;
      2. the environment transitions to its next hidden state;
      3. receives a noisy observation of that new state; and
      4. updates its Bayesian belief.

    The last observation does not reveal the state perfectly.  The terminal
    reward is one exactly when that final, partially observed state is good.
    """
    hidden_state = sample_categorical(PRIOR, rng)
    first_observation = sample_categorical(OBSERVATION[hidden_state], rng)
    belief = incorporate_observation(PRIOR, first_observation)

    observations: list[int] = [first_observation]
    actions: list[int] = []
    beliefs_good: list[float] = [float(belief[GOOD])]

    # A horizon of H has H observations and H - 1 intervening actions.
    for _ in range(horizon - 1):
        action = choose_action(belief)
        actions.append(action)

        hidden_state = sample_categorical(TRANSITION[action, hidden_state], rng)
        observation = sample_categorical(OBSERVATION[hidden_state], rng)

        predicted_belief = predict_next_belief(belief, action)
        belief = incorporate_observation(predicted_belief, observation)
        observations.append(observation)
        # This scalar is enough because P(bad) = 1 - P(good).
        beliefs_good.append(float(belief[GOOD]))

    return {
        "observations": observations,
        "actions": actions,
        "beliefs_good": beliefs_good,
        "terminal_reward": int(hidden_state == GOOD),
    }


def full_history_features(episode: dict, prefix_length: int, horizon: int) -> np.ndarray:
    """Encode the observed prefix in a fixed-width vector.

    Each observation and action is represented as -1 or +1.  Unseen future
    positions are zero-padded.  This gives the history probe access to the
    same evidence used by the Bayesian filter, without handing it the filter's
    known transition and observation probabilities.
    """
    features = np.zeros(2 * horizon, dtype=float)
    observations = np.asarray(episode["observations"][:prefix_length])
    # At prefix length t, only the t - 1 actions that occurred between the
    # observed states are part of the available history.
    actions = np.asarray(episode["actions"][: max(0, prefix_length - 1)])
    features[:prefix_length] = 2.0 * observations - 1.0
    features[horizon : horizon + len(actions)] = 2.0 * actions - 1.0
    return features


def heldout_auc(features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    """Fit on one episode split and report AUC on unseen episodes."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    model = LogisticRegression(max_iter=2_000)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, scores))


def run_experiment(n_episodes: int, horizon: int, seed: int) -> pd.DataFrame:
    """Simulate episodes and evaluate all three representations per prefix."""
    if n_episodes < 100:
        raise ValueError("Use at least 100 episodes for a minimally stable split.")
    if horizon < 2:
        raise ValueError("The horizon must be at least 2.")

    rng = np.random.default_rng(seed)
    episodes = [simulate_episode(horizon, rng) for _ in range(n_episodes)]
    labels = np.asarray([episode["terminal_reward"] for episode in episodes])
    if np.unique(labels).size != 2:
        raise RuntimeError("Simulation produced only one reward class; change the seed.")

    rows = []
    for prefix_length in range(1, horizon + 1):
        # Baseline: only the most recently observed symbol.
        latest_observation = np.asarray(
            [[episode["observations"][prefix_length - 1]] for episode in episodes],
            dtype=float,
        )

        # History representation: every observation and action in the prefix.
        history = np.vstack(
            [
                full_history_features(episode, prefix_length, horizon)
                for episode in episodes
            ]
        )

        # Exact filter: the posterior probability of the good hidden state.
        belief = np.asarray(
            [[episode["beliefs_good"][prefix_length - 1]] for episode in episodes],
            dtype=float,
        )

        rows.append(
            {
                "step": prefix_length,
                "last_observation_auc": heldout_auc(
                    latest_observation, labels, seed
                ),
                "full_history_auc": heldout_auc(history, labels, seed),
                "bayes_belief_auc": heldout_auc(belief, labels, seed),
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-episodes", type=int, default=5_000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/pomdp_results.csv")
    add_wandb_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "n_episodes": args.n_episodes,
        "horizon": args.horizon,
        "seed": args.seed,
    }
    run = init_wandb(args, job_type="pomdp-sanity-check", config=config)

    results = run_experiment(args.n_episodes, args.horizon, args.seed)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved POMDP results -> {output_path}")

    if run is not None:
        # Logging one metric per step makes the three curves directly viewable.
        for row in results.to_dict(orient="records"):
            run.log(row, step=int(row["step"]))
        run.save(str(output_path), base_path=str(output_path.parent))
        run.finish()


if __name__ == "__main__":
    main()
