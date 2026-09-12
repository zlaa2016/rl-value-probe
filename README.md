# RL Value Probe

This repository tests whether an LLM's residual-stream states reveal eventual
task reward before generation is complete. The key comparison is whether those
states predict terminal reward beyond the prompt, generated text, response
length, token identity, verifier progress, and model confidence.

The first experiment evaluated Base, SFT, DPO, and RLVR variants of OLMo 3 on
instruction-following constraints. It produced 120 rollouts and 1,920 saved
activation vectors. The main finding is an identification problem: prompt
identity explained 79.8--100% of reward variance within each model, and only 4
of 24 model-by-prompt cells varied across repeated rollouts. The pilot therefore
validates the extraction pipeline but does not establish an internal value
function.

Read the [project report](rl_probe_project.pdf) for the full experiment,
results, and research motivation.

## What to run

Create the environment and install the IFEval dependency:

```bash
conda env create -f environment.yml
conda activate rl-interp

mkdir -p external
git clone --depth 1 https://github.com/allenai/open-instruct.git external/open-instruct
python -m nltk.downloader punkt punkt_tab
wandb login --verify  # optional
```

Run Experiment 1:

```bash
python run_exp1.py \
  --models base sft dpo rlvr \
  --n-prompts 6 \
  --n-rollouts 5 \
  --max-new-tokens 128 \
  --output-dir outputs/run
```

The runner saves `rollouts.jsonl` and `activations.npz` after every rollout and
supports `--resume`. To analyze a completed run:

```bash
python annotate_constraints.py \
  --rollouts outputs/run/rollouts.jsonl \
  --output outputs/run/rollouts-annotated.jsonl

python evaluate_probes.py \
  --rollouts outputs/run/rollouts-annotated.jsonl \
  --activations outputs/run/activations.npz \
  --output-dir outputs/analysis/run

python audit_evidence.py
```

## Future experiments

The next study should estimate continuation value directly: save an identical
generated prefix, sample many independent continuations, and use their mean
terminal reward as the target. Internal features should then be evaluated on
held-out task families against visible-text, confidence, and verifier
baselines.

- [`docs/experiment_matrix.md`](docs/experiment_matrix.md) lists the recommended
  OLMo pretraining revisions, post-training variants, and Dolci benchmarks.
- [`future_experiments/token_covariance/`](future_experiments/token_covariance/)
  contains the unrun token-wise residual-geometry extraction and analysis code.

## Project structure

| Area | Files |
|---|---|
| Experiment 1 | `run_exp1.py`, `data.py`, `generate.py`, `rewards.py`, `activations.py` |
| Analysis | `evaluate_probes.py`, `audit_evidence.py`, `probe.py`, `analysis_utils.py` |
| Utilities | `tracking.py`, `download_wandb_artifact.py`, `merge_outputs.py`, `annotate_constraints.py` |
| Future work | `docs/experiment_matrix.md`, `future_experiments/` |
| Report | `rl_probe_project.pdf` |

Secrets are read from `wandb login`, `WANDB_API_KEY`, `hf auth login`, or
`HF_TOKEN`; they are not stored in the repository.

## Tests

```bash
python -m unittest -v test_*.py
```
