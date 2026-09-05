# RL Value Probe

This project asks whether an LLM's intermediate hidden state contains a
trajectory-specific estimate of eventual instruction-following reward, and
whether that relationship changes from Base through SFT, DPO, and RLVR.

The completed pilot does **not** establish that claim. Its most important result
is that ordinary probe performance would be misleading on the collected data:

- prompt identity accounts for 79.8--100% of reward variance within checkpoint;
- only 4 of 24 checkpoint-by-prompt cells vary across stochastic rollouts;
- both held-out prompts have constant zero reward, so held-out R2 is undefined;
- pooled confidence and activation correlations change substantially after
  prompt centering; and
- every response reaches the 128-token cutoff, so the target is reward on a
  truncated prefix rather than on a naturally completed response.

A high-dimensional probe can therefore succeed by identifying the prompt or
visible constraint progress. The scientific question is not whether reward can
be fitted, but whether activations contain information about future reward
*beyond* those observable explanations.

## Completed experiment

Four matched OLMo 3 checkpoints were evaluated:

| Stage | Hugging Face checkpoint |
|---|---|
| Base | `allenai/Olmo-3-1025-7B` |
| SFT | `allenai/Olmo-3-7B-Think-SFT` |
| DPO | `allenai/Olmo-3-7B-Think-DPO` |
| RLVR | `allenai/Olmo-3-7B-Think` |

The experiment used six prompts, five sampled responses per prompt and stage,
and four activation snapshots per response. This produced 120 rollouts and
1,920 residual-stream vectors of width 4,096.

Prompts came from the pinned revision in `config.py` of
`allenai/Dolci-Think-RL-7B`, filtered to
`hamishivi/IF_multi_constraints_upto5_filtered`. IFEvalG assigns one terminal
score equal to the fraction of explicit constraints satisfied. This is a
formatting and lexical-compliance reward, not a general correctness score.

The completed runs are stored in the W&B project
`zl6113-new-york-university/rl-interp`:

| Stage | Run ID |
|---|---|
| Base | `hmmzv6og` |
| SFT | `s1zmpbyt` |
| DPO | `qoevdhnj` |
| RLVR | `dti13bc0` |

## Analysis narrative

`evaluate_probes.py` implements Ridge probes over current activations,
activation history, activation changes, policy confidence, and combined
features. Splits and shuffled-label nulls are grouped by prompt.

`audit_evidence.py` performs the more important identification checks. It
decomposes reward variance by prompt, recomputes correlations after prompt
centering, measures leave-one-prompt-out sensitivity, applies the verifier to
partial text, and estimates checkpoint differences at the prompt level. These
checks show why a probe score alone cannot support a value-representation
claim.

The next decisive experiment is one design, not another probe family: save many
prefix states and sample multiple independent continuations from each identical
prefix. Their average terminal score estimates

```text
V(prefix) = E[terminal reward | prompt and generated prefix].
```

Activations become informative only if they predict this continuation value on
held-out constraint families and improve over policy confidence and current
partial-verifier status. Cross-checkpoint transfer and activation intervention
then test separate questions about post-training and causal use.

## Reproduce the workflow

Create the environment and lightweight IFEval dependency:

```bash
conda env create -f environment.yml
conda activate rl-interp

mkdir -p external
git clone --depth 1 https://github.com/allenai/open-instruct.git external/open-instruct
python -m nltk.downloader punkt punkt_tab
wandb login --verify
```

Generate a complete experiment:

```bash
python run_mvp.py \
  --models base sft dpo rlvr \
  --n-prompts 6 \
  --n-rollouts 5 \
  --max-new-tokens 128 \
  --output-dir outputs/run
```

The runner checkpoints `rollouts.jsonl` and `activations.npz` after every
rollout and supports `--resume`. W&B uploads both artifacts after each stage.

Analyze the completed local artifacts:

```bash
python annotate_constraints.py \
  --rollouts outputs/all-models-six-5ro/rollouts.jsonl \
  --output outputs/all-models-six-5ro/rollouts-annotated.jsonl

python evaluate_probes.py \
  --rollouts outputs/all-models-six-5ro/rollouts-annotated.jsonl \
  --activations outputs/all-models-six-5ro/activations.npz \
  --output-dir outputs/analysis/all-models-six-5ro \
  --n-label-shuffles 200 \
  --wandb-mode online

python audit_evidence.py
python make_paper_figures.py
```

`paper.tex` is the canonical manuscript. `paper_no_figures.tex` is a small
entry point that compiles the same source with figures disabled. Both use
`references.bib`.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error paper_no_figures.tex
```

## Project structure

| Area | Files |
|---|---|
| Rollout pipeline | `run_mvp.py`, `data.py`, `generate.py`, `rewards.py`, `activations.py` |
| Analysis | `evaluate_probes.py`, `audit_evidence.py`, `probe.py`, `analysis_utils.py` |
| Tracking and artifact utilities | `tracking.py`, `download_wandb_artifact.py`, `merge_outputs.py`, `annotate_constraints.py` |
| Paper | `paper.tex`, `paper_no_figures.tex`, `references.bib`, `make_paper_figures.py`, `figures/` |

Secrets are read from `wandb login`, `WANDB_API_KEY`, `hf auth login`, or
`HF_TOKEN`; they are not stored in the repository.

## Tests

```bash
python -m unittest -v test_*.py
```

The current suite contains 19 tests.
