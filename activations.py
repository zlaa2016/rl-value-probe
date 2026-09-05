import math
import numpy as np
import torch


def choose_layers(model):
    """
    Four evenly spaced transformer block outputs.
    hidden_states[0] is the embedding output, so block outputs are 1..L.
    """
    n_layers = int(model.config.num_hidden_layers)
    return sorted({
        max(1, n_layers // 4),
        max(1, n_layers // 2),
        max(1, 3 * n_layers // 4),
        n_layers,
    })


def _token_confidence_signals(model, final_hidden, generated_ids, chunk_size=64):
    """
    Compute token-level confidence from the policy itself.

    final_hidden: hidden states whose positions predict the generated tokens,
                  shape [T, d_model].
    generated_ids: sampled generated tokens, shape [T].

    Returns arrays for:
      - chosen-token log probability
      - entropy of next-token distribution
      - top-1 probability

    We apply the LM head in chunks to avoid materializing a huge
    [sequence_length, vocab_size] tensor all at once.
    """
    chosen_logprobs = []
    entropies = []
    top1_probs = []

    for start in range(0, len(generated_ids), chunk_size):
        end = min(len(generated_ids), start + chunk_size)

        h = final_hidden[start:end]
        target = generated_ids[start:end].to(h.device)

        logits = model.lm_head(h).float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        chosen = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        top1 = probs.max(dim=-1).values

        chosen_logprobs.append(chosen.cpu())
        entropies.append(entropy.cpu())
        top1_probs.append(top1.cpu())

        del logits, log_probs, probs

    return {
        "token_logprob": torch.cat(chosen_logprobs).numpy(),
        "token_entropy": torch.cat(entropies).numpy(),
        "top1_prob": torch.cat(top1_probs).numpy(),
    }


@torch.inference_mode()
def extract_states_and_signals(
    model,
    prompt_ids,
    generated_ids,
    fractions=(0.25, 0.50, 0.75, 1.00),
    layers=None,
):
    """
    Re-run the finished sequence once.

    Extract:
      1. selected hidden states for activation probes
      2. policy confidence signals (generation probability / entropy)

    For generated token y_j, logits at the previous sequence position
    predict y_j. We therefore take final hidden states from:
        prompt_len - 1 ... prompt_len + gen_len - 2
    and apply the model's LM head.
    """
    if len(generated_ids) == 0:
        raise ValueError("Cannot extract trajectory states from an empty generation.")

    if layers is None:
        layers = choose_layers(model)

    full_ids = torch.cat([prompt_ids, generated_ids]).unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(full_ids)

    # We only need hidden states; setting logits_to_keep is model-specific,
    # so keep this generic and use the returned hidden states directly.
    out = model(
        input_ids=full_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    prompt_len = len(prompt_ids)
    gen_len = len(generated_ids)

    vectors = {}
    token_positions = {}

    for frac in fractions:
        offset = round(frac * gen_len) - 1
        offset = max(0, min(gen_len - 1, offset))
        token_idx = prompt_len + offset

        token_positions[frac] = int(offset)

        for layer in layers:
            vec = out.hidden_states[layer][0, token_idx]
            vectors[(layer, frac)] = vec.float().cpu().numpy()

    # Hidden position i predicts token i+1.
    pred_hidden = out.hidden_states[-1][
        0,
        prompt_len - 1 : prompt_len + gen_len - 1,
    ]

    token_signals = _token_confidence_signals(
        model=model,
        final_hidden=pred_hidden,
        generated_ids=generated_ids,
    )

    prefix_signals = {}
    for frac in fractions:
        end = max(1, min(gen_len, round(frac * gen_len)))

        lp = token_signals["token_logprob"][:end]
        ent = token_signals["token_entropy"][:end]
        top1 = token_signals["top1_prob"][:end]

        mean_lp = float(lp.mean())

        prefix_signals[f"{frac:.2f}"] = {
            # Stable version of "generation probability".
            "mean_token_logprob": mean_lp,
            "geomean_token_prob": float(math.exp(mean_lp)),
            # Raw sum is included but is strongly length-dependent.
            "sequence_logprob": float(lp.sum()),
            "mean_entropy": float(ent.mean()),
            "mean_top1_prob": float(top1.mean()),
            "last_token_logprob": float(lp[-1]),
            "last_token_entropy": float(ent[-1]),
            "last_top1_prob": float(top1[-1]),
        }

    return vectors, token_positions, prefix_signals
