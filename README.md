# RL Value Probe

This repository measures how well terminal instruction-following reward can be
decoded from intermediate hidden states across four OLMo 3 checkpoints:

| Stage | Hugging Face model |
|---|---|
| Base | `allenai/Olmo-3-1025-7B` |
| SFT | `allenai/Olmo-3-7B-Think-SFT` |
| DPO | `allenai/Olmo-3-7B-Think-DPO` |
| RLVR | `allenai/Olmo-3-7B-Think` |

The Base checkpoint is the model baseline. Shuffled terminal-reward labels are
the statistical null. There is no TF-IDF baseline.

## Data and reward

`data.py` streams the pinned revision
`0fb6466d31ef3a9dd16985ef635e6429e05a6491` of
`allenai/Dolci-Think-RL-7B` and selects rows whose `original_dataset` is
`hamishivi/IF_multi_constraints_upto5_filtered`.

Each model generates a complete response for each prompt. `rewards.py` evaluates
the response with the IFEvalG constraint registry from Ai2's `open-instruct`
source tree. The terminal reward is the fraction of the prompt's constraints
that passed.

Install the verifier source without installing the full `open-instruct`
training package:

```bash
mkdir -p external
git clone --depth 1 https://github.com/allenai/open-instruct.git external/open-instruct
python -m nltk.downloader punkt punkt_tab
```

## What is recorded

`run_mvp.py` generates rollouts and saves:

- model stage, model name, prompt ID, and rollout index;
- generated text and generated token IDs;
- terminal reward;
- partial text at 25%, 50%, 75%, and 100% of the generated response;
- policy confidence at the same four fractions;
- selected hidden-state vectors and their token positions.

Policy confidence contains the mean and last-token values for chosen-token log
probability, next-token entropy, and top-1 probability.

For each trajectory fraction, the activation is the residual-stream hidden
vector at the generated token nearest that fraction. It is not mean pooling over
the response. Four transformer layers are recorded: 8, 16, 24, and 32.

The output files are:

```text
<output-dir>/
├── rollouts.jsonl
└── activations.npz
```

Both files are checkpointed after every rollout. `--resume` skips existing
`model_stage/prompt_id/rollout_index` combinations. W&B uploads the current
files after each completed model stage.

## Completed six-prompt run

The completed run used six prompts, five sampled responses per prompt, and 128
generated tokens per response.

| Stage | W&B run name | Run ID | State |
|---|---|---|---|
| Base | `base-six-promt-5ro` | `hmmzv6og` | finished |
| SFT | `sft-six-promt-5ro` | `s1zmpbyt` | finished |
| DPO | `dpo-six-promt-5ro` | `qoevdhnj` | finished |
| RLVR | `rlvr-six-promt-5ro` | `dti13bc0` | finished |

W&B project: `zl6113-new-york-university/rl-interp`.

The downloaded Base directory contains both the 30 Base rollouts and an exact
copy of the 30 SFT rollouts. The standalone SFT download duplicates those SFT
rollout IDs, so it is intentionally excluded from the merge command:

```bash
python merge_outputs.py \
  outputs/raw/base-six-promt-5ro \
  outputs/raw/dpo-six-promt-5ro \
  outputs/raw/rlvr-six-promt-5ro \
  --output-dir outputs/all-models-six-5ro
```

The merged local dataset contains:

```text
120 rollouts total
30 rollouts per model stage
6 unique prompts
5 rollouts per prompt and stage
1,920 activation rows
4,096 values per activation vector
4 layers × 4 trajectory fractions per rollout
```

The merged files are in `outputs/all-models-six-5ro/`.

Publication-ready summary figures are tracked in `figures/`.

## Paper and figures

`paper.tex` is the results-grounded manuscript for the completed run. Its
first page is an executive summary, followed by Introduction, Methods, Results,
Discussion, Limitations, Conclusion, and an appendix with a worked prompt
example.

Regenerate all paper figures from the saved rollout and analysis artifacts:

```bash
python make_paper_figures.py
```

The manuscript places the generated PNG files as follows:

| PNG | Placement in `paper.tex` |
|---|---|
| `figures/executive_summary.png` | Executive Summary, first page |
| `figures/constraint_pass_rates.png` | Results: constraint outcomes |
| `figures/trajectory_correlation_with_terminal_reward.png` | Results: trajectory correlations |
| `figures/example_prompt_639_88.png` | Appendix: example prompt |

Compile locally with a LaTeX distribution that provides `acmart`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

## Probe evaluation

`evaluate_probes.py` makes one fixed train/test split grouped by prompt. With
the completed six-prompt dataset, four prompt groups are used for training and
two are held out. All five rollouts from a prompt remain in the same split.

For every checkpoint, layer, and trajectory fraction, it evaluates Ridge
regression using:

- the current activation;
- the concatenated activation history up to the current fraction;
- the activation change from the preceding fraction;
- policy-confidence features alone;
- the current activation combined with policy confidence.

Each probe predicts the rollout's terminal reward. The shuffled-label null
permutes complete prompt blocks so related rollouts remain grouped. Reported
statistics include held-out R2, MSE, a rank-correlation measure, null summaries,
and permutation p-values. R2 and rank correlation are reported as undefined
when the held-out rewards have zero variance; MSE and target variance remain in
the output.

Run the completed dataset analysis with:

```bash
python evaluate_probes.py \
  --rollouts outputs/all-models-six-5ro/rollouts.jsonl \
  --activations outputs/all-models-six-5ro/activations.npz \
  --output-dir outputs/analysis/all-models-six-5ro \
  --test-size 0.25 \
  --ridge-alpha 10 \
  --n-label-shuffles 200 \
  --seed 0 \
  --wandb-run-name all-models-six-5ro-analysis \
  --wandb-mode online
```

The analysis writes:

```text
probe_metrics.csv
heldout_predictions.csv
instruction_trajectories.csv
faithfulness_correlations.csv
trajectory_correlation_with_terminal_reward.png
constraint_difficulty.csv              # when constraint annotations exist
constraint_correlations.csv            # when constraint annotations exist
constraint_pass_rates.png              # when constraint annotations exist
instruction_trajectories/*.png
heldout_probe_trajectories/*.png
```

It also logs the tables, final-layer held-out R2 curves, and trajectory images
to W&B and uploads the analysis directory as an artifact.

The completed rollout files predate storage of `prompt_text` and individual
constraint results. Create an annotated JSONL copy before analysis when those
fields are needed:

```bash
python annotate_constraints.py \
  --rollouts outputs/all-models-six-5ro/rollouts.jsonl \
  --output outputs/all-models-six-5ro/rollouts-annotated.jsonl
```

Use `rollouts-annotated.jsonl` as the `--rollouts` argument to obtain labeled
instruction plots and constraint-level tables. The activation file is unchanged.

## Setup and authentication

```bash
conda env create -f environment.yml
conda activate rl-interp
wandb login --verify
```

Alternatively, install into an existing Python 3.11 environment:

```bash
pip install -r requirements.txt
```

W&B authentication comes from `wandb login` or the `WANDB_API_KEY` environment
variable. Hugging Face authentication comes from `hf auth login` or `HF_TOKEN`.
Keys are not stored in the repository.

## Main files

| File | Purpose |
|---|---|
| `run_mvp.py` | Generate, score, extract, checkpoint, and log rollouts |
| `data.py` | Load the pinned instruction-following prompt source |
| `generate.py` | Load checkpoints and sample responses |
| `rewards.py` | Compute aggregate and per-constraint IFEval rewards |
| `activations.py` | Extract hidden vectors and policy-confidence signals |
| `merge_outputs.py` | Merge rollout directories with duplicate validation |
| `annotate_constraints.py` | Add prompt text and per-constraint results to older runs |
| `evaluate_probes.py` | Run held-out probes, null tests, correlations, and plots |
| `make_paper_figures.py` | Regenerate the four manuscript figures from saved outputs |
| `paper.tex` | Results-grounded manuscript and appendix |
| `paper_no_figures.tex` | Standalone manuscript version with figure blocks removed |
| `references.bib` | BibTeX references used by both manuscript versions |
| `tracking.py` | Log metrics, tables, plots, and artifacts to W&B |

## Scope of the completed run

The completed dataset measures reward decodability and within-response
trajectories on six prompts drawn from the RLVR training dataset. Its held-out
split is a probe evaluation over two prompt groups from that same source. It
does not constitute an unseen-dataset generalization or memorization test.

## Tests

```bash
python -m unittest -v \
  test_activations.py \
  test_rewards.py \
  test_probe.py \
  test_run_mvp.py \
  test_merge_outputs.py \
  test_evaluate_probes.py
```

The current suite contains 17 passing tests.
