# RL Value Probe MVP

A small experiment for asking:

> How does terminal-reward decodability from an LLM's intermediate hidden states change across Base -> SFT -> DPO -> RLVR?

The main experiment compares matched Olmo 3 post-training stages:

- `allenai/Olmo-3-1025-7B` (base)
- `allenai/Olmo-3-7B-Think-SFT`
- `allenai/Olmo-3-7B-Think-DPO`
- `allenai/Olmo-3-7B-Think` (final RLVR)

It uses prompts from `allenai/Dolci-Think-RL-7B`, initially the **IF RLVR Mixture**, because it has a terminal instruction-following verifier reward.

## Project structure

```text
rl_value_probe_mvp/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── data.py
├── generate.py
├── rewards.py
├── activations.py
├── baselines.py
├── probe.py
├── pomdp.py
├── run_mvp.py
└── outputs/
    └── .gitkeep
```

## Research question

For hidden state `h_t` at rollout time `t` and terminal reward `R_T`:

```text
activation monitor:   h_t     -> R_T
training comparison:  Base -> SFT -> DPO -> RLVR
```

The first MVP asks whether terminal reward becomes linearly decodable from hidden states as the rollout unfolds, and whether this changes across Base -> SFT -> DPO -> RLVR.

This is **not** claiming that the model has learned a literal value function. The probe is a post-hoc estimator:

`V_hat(h_t) ~= E[R_T | h_t]`.

## Why the POMDP comparison is mathematically sound

A true latent state in an MDP is Markov:

`P(s_{t+1} | s_0,...,s_t,a_t) = P(s_{t+1} | s_t,a_t)`.

But under partial observability, the agent sees observations, not `s_t`. The observable history can therefore remain predictive. In a POMDP, the Bayesian belief

`b_t(s) = P(s_t=s | observation/action history)`

is a sufficient state representation.

The controlled POMDP comparison is:

```text
POMDP:   history -> exact Bayesian belief b_t -> future outcome
LLM:     learned hidden state h_t -> terminal reward
```

The included `pomdp.py` is a small controlled sanity check. It compares terminal-outcome prediction from:

1. the latest observation only,
2. the full observable history,
3. the exact Bayesian belief.

If belief is sufficient, full history should add little once belief is known. For the LLM, the scientific question is whether post-training makes terminal outcome increasingly decodable from `h_t`.

## Setup

Use the project environment `rl-interp`:

```bash
conda env create -f environment.yml
conda activate rl-interp
```

Or create it manually:

```bash
conda create -n rl-interp python=3.11 -y
conda activate rl-interp
pip install -r requirements.txt
```

### Exact IF reward verifier

The MVP uses Ai2's IFEvalG constraint registry from `open-instruct`. Only the
source tree is needed; do not install the full `open-instruct` package, whose
training stack currently requires Python 3.12 and is unnecessary here.

```bash
mkdir -p external
git clone --depth 1 https://github.com/allenai/open-instruct.git external/open-instruct
python -m nltk.downloader punkt punkt_tab
```

`rewards.py` imports only the verifier registry from that checkout. The small
runtime dependencies are included in this project's `requirements.txt`.

## GPU / compute

Recommended pilot:

- 1 GPU
- 24 GB VRAM minimum for careful batch-1 BF16 inference
- 40-80 GB is more comfortable
- no RL training required

Start very small:

```bash
python run_mvp.py --models base sft --n-prompts 5 --n-rollouts 2
```

`base sft` is also the default model selection, so the pretrained base-model
baseline is always present in a plain `python run_mvp.py` pilot.

Then:

```bash
python probe.py
python pomdp.py
```

## Weights & Biases tracking

The scripts log to `zl6113-new-york-university/rl-interp` by default. Authenticate
once on each machine before an online run:

```bash
wandb login --verify
```

Do not put the API key in source code. In Kaggle, create a secret named
`WANDB_API_KEY`; locally, `wandb login` is preferred. You can override the
destination with `WANDB_ENTITY` and `WANDB_PROJECT` environment variables or
the matching command-line flags.

`run_mvp.py` logs live rollout rewards and a trajectory table containing the
partial text and policy-confidence signals at 25%, 50%, 75%, and 100%. It also
creates trajectory charts comparing model stages. `probe.py` logs the full
probe-results table and the pre-specified final-layer held-out probe R2 at each
trajectory fraction. It also uploads the saved rollout trajectories, so outputs produced by
an already-running older process can be visualized without regeneration.
`pomdp.py` logs its three AUC curves.

For a local run that can be synced later:

```bash
python run_mvp.py --models base sft --n-prompts 5 --n-rollouts 2 --wandb-mode offline
wandb sync wandb/offline-run-*
```

To turn tracking off entirely, add `--wandb-mode disabled`.

Full MVP:

```bash
python run_mvp.py \
  --models base sft dpo rlvr \
  --n-prompts 40 \
  --n-rollouts 4 \
  --max-new-tokens 512

python probe.py
```

The script loads one model at a time, so you do not need memory for all four simultaneously.
It prints the parameter placements after each load (for example `mps`, CUDA
devices, CPU, or disk), making hardware use and offloading explicit.

## Outputs

`run_mvp.py` creates:

```text
outputs/
├── rollouts.jsonl
└── activations.npz
```

Each rollout stores:

- model stage
- prompt id
- generated text
- terminal reward
- partial text at 25%, 50%, 75%, 100%

Both output files are atomically checkpointed after every completed rollout.
When W&B is enabled, they are also uploaded after each completed model stage,
so a later Kaggle interruption does not erase all earlier-stage results.
To extend an interrupted or smaller run, pass `--resume` with the same output
directory and set `--n-rollouts` to the desired total per prompt. Existing
stage/prompt/rollout-index combinations are retained and skipped.

Runs produced on separate machines can be combined before probing:

```bash
python merge_outputs.py outputs/local-base outputs/kaggle-posttrained \
  --output-dir outputs/all-models
```

The NPZ stores hidden vectors indexed by rollout, layer, and trajectory fraction.

`probe.py` writes:

```text
outputs/probe_results.csv
```

For each model stage it reports:

- ridge regression on activations
- ridge regression on within-rollout activation changes between consecutive fractions
- a separate ridge diagnostic on policy-confidence signals
- a shuffled-training-label null distribution and permutation p-values

All train/test splits are grouped by prompt to avoid putting different rollouts of the same prompt in train and test.
The default null uses 50 label shuffles; increase it for final analyses with
`python probe.py --n-label-shuffles 200`.

## Push to GitHub

Create an empty repository on GitHub first, then:

```bash
git init
git add .
git commit -m "Initial RL value-probe MVP"
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/rl-value-probe.git
git push -u origin main
```

HTTPS alternative:

```bash
git remote add origin https://github.com/YOUR_USERNAME/rl-value-probe.git
git push -u origin main
```


## Policy-confidence rewards / baselines

The MVP now also records three model-internal confidence families discussed in
recent RL-for-reasoning work:

- **generation probability**: chosen-token log-probability / geometric mean probability
- **entropy**: uncertainty of the next-token distribution
- **top-1 confidence**: maximum next-token probability

These are **proxies**, not correctness labels. A model can be confidently wrong,
miscalibrated, or become more confident because RL sharpens its distribution.

For each rollout prefix (25/50/75/100%), `run_mvp.py` stores these signals in
`outputs/rollouts.jsonl`. `probe.py` fits a small ridge model on the training
prompts and evaluates it only on held-out prompts.

The activation probe and confidence baseline are therefore evaluated on the
same held-out prompt split.

Cross-attention is **not** included in the MVP. OLMo is a decoder-only language
model, so a generic "cross-attention reward" is not directly available in the
same sense as in architectures with separate encoder/context streams. Add this
only if reproducing a specific paper's definition.

## Suggested experiment order

1. `pomdp.py` — verify the theory/sanity check locally, CPU only.
2. 5 prompts x 2 rollouts x Base/SFT — verify generation, reward, activation extraction.
3. 20 prompts x 4 rollouts x one model — make sure probe evaluation works.
4. 40-100 prompts x 4 rollouts x Base/SFT/DPO/RLVR.
5. Only then add intermediate RL checkpoints, more complex history models, activation patching, or MemoryArena.

## Interpretation

Interesting result:

- terminal reward becomes decodable earlier in the trajectory;
- and/or this early decodability changes systematically from Base through RLVR.

Null result is also informative:

- post-training does not systematically improve held-out terminal-reward decodability.

Do not interpret probe success alone as causal evidence or as proof of an explicit learned value function.
