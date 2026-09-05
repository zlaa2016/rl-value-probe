"""
A tiny controlled POMDP sanity check.

Question:
If the latent process is Markov, can observable history still predict
terminal outcome?

Yes. In a POMDP, observations are only partial/noisy views of the latent
Markov state. History helps infer the current belief state.

Here:
- hidden state s_t in {bad=0, good=1}
- action 0 tends to preserve state
- action 1 tends to flip state
- observation is a noisy readout of the hidden state
- policy chooses preserve when belief(good) >= .5, otherwise flip
- terminal reward is 1 iff final hidden state is good

We compare terminal-reward prediction from:
1) latest observation only
2) full observed history
3) exact Bayesian belief b_t = P(s_t | history)

The belief is the mathematically privileged compressed state.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tracking import add_wandb_args, init_wandb


TRANSITION = np.array([
    # action 0: preserve with probability .90
    [[0.90, 0.10],
     [0.10, 0.90]],

    # action 1: flip with probability .90
    [[0.10, 0.90],
     [0.90, 0.10]],
], dtype=float)

# OBS[s, o] = P(observation=o | hidden state=s)
OBS = np.array([
    [0.75, 0.25],
    [0.25, 0.75],
], dtype=float)

PRIOR = np.array([0.50, 0.50], dtype=float)


def belief_update(belief, action, observation):
    """
    Bayesian filtering:
      prediction(s') = sum_s T[a,s,s'] * b(s)
      posterior(s') proportional to O[s',o] * prediction(s')
    """
    predicted = belief @ TRANSITION[action]
    posterior = predicted * OBS[:, observation]
    posterior /= posterior.sum()
    return posterior


def policy(belief):
    """Try to finish in state 'good'."""
    return 0 if belief[1] >= 0.5 else 1


def sample_categorical(rng, probs):
    return int(rng.choice(len(probs), p=probs))


def simulate_episode(rng, horizon=8):
    state = sample_categorical(rng, PRIOR)
    obs = sample_categorical(rng, OBS[state])
    belief = PRIOR * OBS[:, obs]
    belief /= belief.sum()

    observations = [obs]
    actions = []
    beliefs = [belief.copy()]

    for _ in range(horizon - 1):
        action = policy(belief)
        actions.append(action)

        state = sample_categorical(rng, TRANSITION[action, state])
        obs = sample_categorical(rng, OBS[state])
        observations.append(obs)

        belief = belief_update(belief, action, obs)
        beliefs.append(belief.copy())

    reward = int(state == 1)

    return {
        "observations": observations,
        "actions": actions,
        "beliefs": beliefs,
        "reward": reward,
    }


def prefix_features(ep, t, horizon):
    """
    t is zero-indexed current observation position.
    History vector uses observations/actions up to t and pads the future with -1.
    """
    obs_hist = ep["observations"][:t + 1]
    act_hist = ep["actions"][:t]

    obs_pad = obs_hist + [-1] * (horizon - len(obs_hist))
    act_pad = act_hist + [-1] * ((horizon - 1) - len(act_hist))

    return np.array(obs_pad + act_pad, dtype=float)


def auc_for_features(X, y, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, pred)


def run_experiment(n_episodes=5000, horizon=8, seed=0):
    rng = np.random.default_rng(seed)
    episodes = [simulate_episode(rng, horizon) for _ in range(n_episodes)]
    y = np.array([ep["reward"] for ep in episodes])

    rows = []

    for t in range(horizon):
        last_obs = np.array(
            [[ep["observations"][t]] for ep in episodes],
            dtype=float,
        )

        history = np.stack([
            prefix_features(ep, t, horizon)
            for ep in episodes
        ])

        belief = np.array([
            ep["beliefs"][t]
            for ep in episodes
        ])

        rows.append({
            "t": t + 1,
            "last_observation_auc": auc_for_features(last_obs, y, seed),
            "full_history_auc": auc_for_features(history, y, seed),
            "bayes_belief_auc": auc_for_features(belief, y, seed),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/pomdp_results.csv")
    add_wandb_args(parser)
    args = parser.parse_args()

    run = init_wandb(
        args,
        job_type="pomdp-sanity-check",
        config={
            "n_episodes": args.n_episodes,
            "horizon": args.horizon,
            "seed": args.seed,
        },
    )

    df = run_experiment(
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        seed=args.seed,
    )

    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    if run is not None:
        import wandb

        run.log({
            "pomdp/results": wandb.Table(dataframe=df),
            "pomdp/auc_over_time": wandb.plot.line_series(
                xs=df["t"].tolist(),
                ys=[
                    df["last_observation_auc"].tolist(),
                    df["full_history_auc"].tolist(),
                    df["bayes_belief_auc"].tolist(),
                ],
                keys=["last observation", "full history", "Bayes belief"],
                title="Terminal-reward prediction in the POMDP",
                xname="trajectory step",
            ),
        })
        run.finish()
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
