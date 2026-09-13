# RL Value Probe

This repository tests whether an LLM's residual-stream states reveal eventual
task reward before generation is complete. The key comparison is whether those
states predict terminal reward beyond the prompt, generated text, response
length, token identity, verifier progress, and model confidence.

Read the [project report](rl_probe_project.pdf) for the full experiment,
results, and research motivation.

## Data used

- Dataset: `allenai/Dolci-Think-RL-7B`, pinned to revision
  `0fb6466d31ef3a9dd16985ef635e6429e05a6491`.
- Subset: `hamishivi/IF_multi_constraints_upto5_filtered`, containing prompts
  with multiple explicit instruction-following constraints.
- Sample: 6 prompts, 5 stochastic rollouts per prompt, and 4 model variants,
  for 120 generated responses.
- Models: `allenai/Olmo-3-1025-7B`, `allenai/Olmo-3-7B-Think-SFT`,
  `allenai/Olmo-3-7B-Think-DPO`, and `allenai/Olmo-3-7B-Think`.
- Terminal reward: fraction of IFEval constraints satisfied by the generated
  response.
- Saved measurements: generated text and token IDs, reward and per-constraint
  outcomes, model-confidence statistics, and residual-stream states at four
  generated-token fractions and four transformer layers.

## Other OLMo 3 RLVR data and checkpoints

OLMo 3 also provides controlled RL-Zero models trained directly from the 7B
base model. These are useful for isolating the effect of one reward domain:

| Domain | Model | RLVR dataset |
|---|---|---|
| Math | `allenai/Olmo-3-7B-RL-Zero-Math` | `allenai/Dolci-RL-Zero-Math-7B` |
| Code | `allenai/Olmo-3-7B-RL-Zero-Code` | `allenai/Dolci-RL-Zero-Code-7B` |
| Instruction following | `allenai/Olmo-3-7B-RL-Zero-IF` | `allenai/Dolci-RL-Zero-IF-7B` |
| General | `allenai/Olmo-3-7B-RL-Zero-General` | `allenai/Dolci-RL-Zero-General-7B` |
| Mixed domains | `allenai/Olmo-3-7B-RL-Zero-Mix` | `allenai/Dolci-RL-Zero-Mix-7B` |

The broader released family also has 32B Think and 7B/32B Instruct tracks,
each with SFT, DPO, and final RLVR variants. For within-base training-time
comparisons, `allenai/Olmo-3-1025-7B` exposes intermediate revisions. The
recommended sparse panel is:

- `stage1-step10000`
- `stage1-step700000`
- `stage1-step1413814`
- `stage2-step47684`
- `main`

See [`docs/experiment_matrix.md`](docs/experiment_matrix.md) for the larger
Dolci source inventory, row counts, scorer requirements, and additional
checkpoint suggestions.

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
