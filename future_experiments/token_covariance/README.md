# Token-wise residual geometry (future experiment)

This folder preserves the proposed token-wise covariance experiment separately
from the current four-snapshot probe pipeline.

## Object being measured

For one rollout and one transformer layer, save the residual-stream state at
every generated-token position:

```text
H.shape == [generated_tokens, hidden_width]
```

For the OLMo checkpoints in this repository, `hidden_width` is 4096. A
128-token rollout therefore has `H.shape == [128, 4096]`.

`extract_full_states.py` provides a replay helper that collects this matrix at
each selected layer. The functions in `token_covariance.py` compute:

- `feature_covariance(H)`: a `[4096, 4096]` covariance of residual dimensions
  across token positions;
- `token_gram(H)`: a `[T, T]` centered similarity matrix between contextual
  token-position states;
- `token_covariance(H)`: a `[T, T]` coordinate-wise covariance between
  token-position states; and
- `spectral_summary(H)`: compact eigenvalue/trace/effective-rank features that
  avoid flattening a huge covariance matrix.

The token matrices are **not attention matrices**. Attention records how query
positions weight key positions. These matrices record similarity/covariance
between residual-stream vectors.

## Causal-time rule

At prediction time `t`, compute every feature using only `H[:t]`. Never use a
state from a future token. `prefix_summaries` enforces this and produces one
row per requested prefix length.

Run candidate predictors separately (entropy, self-certainty, activation,
token geometry) before testing combined models. The terminal IFEval constraint
outcome remains the dependent variable. A covariance statistic is a feature;
it becomes a reward predictor only after a mapping from that feature to the
terminal outcome is specified and evaluated on held-out prompt families.

## Example

Use `extract_full_generated_states(...)` after generation to save a complete
rollout. Its `states` array has shape `[layers, generated_tokens, hidden_width]`.
Select one layer (for example, `payload["states"][2]`) and save that matrix as a
NumPy `.npy` file, then run:

```bash
python future_experiments/token_covariance/token_covariance.py \
  --activations full_token_states_layer32.npy \
  --out outputs/future/token_geometry_layer32.npz \
  --prefix-lengths 16 32 64 128 \
  --top-k 8
```

The output contains the full-sequence token Gram/covariance matrices and
compact causal-prefix summaries. Full feature covariance is omitted by default
because one float32 `[4096, 4096]` matrix is about 64 MiB; pass
`--save-feature-covariance` only when it is genuinely needed.

## Data still needed

The existing `outputs/all-models-six-5ro/activations.npz` cannot support this
experiment: it contains only four positions per rollout. A future generation
run must call the provided extraction helper to save every generated-token
residual state at each selected layer. It is intentionally not wired into the
current pilot, so it cannot silently increase that run's storage requirements.
