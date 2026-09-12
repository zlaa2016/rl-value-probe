"""Collect every generated-token residual state for the future experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def choose_layers(model) -> list[int]:
    """Choose four roughly evenly spaced transformer block outputs."""
    count = int(model.config.num_hidden_layers)
    return sorted(
        {
            max(1, count // 4),
            max(1, count // 2),
            max(1, 3 * count // 4),
            count,
        }
    )


@torch.inference_mode()
def extract_full_generated_states(
    model,
    prompt_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    layers: list[int] | tuple[int, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Replay a rollout and return states with shape [layers, tokens, width].

    Row ``t`` is the contextual residual-stream state at the position occupied
    by generated token ``t``. It therefore represents that token plus its left
    context; it is not the state that predicted the token.
    """
    prompt_ids = prompt_ids.reshape(-1)
    generated_ids = generated_ids.reshape(-1)
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids cannot be empty")
    if len(generated_ids) == 0:
        raise ValueError("generated_ids cannot be empty")

    selected_layers = choose_layers(model) if layers is None else list(layers)
    layer_count = int(model.config.num_hidden_layers)
    if not selected_layers or any(
        layer < 1 or layer > layer_count for layer in selected_layers
    ):
        raise ValueError(f"layers must be transformer blocks in 1..{layer_count}")
    if len(set(selected_layers)) != len(selected_layers):
        raise ValueError("layers must be unique")

    full_ids = torch.cat([prompt_ids, generated_ids]).unsqueeze(0).to(model.device)
    output = model(
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        output_hidden_states=True,
        use_cache=False,
    )

    start = len(prompt_ids)
    stop = start + len(generated_ids)
    states = torch.stack(
        [output.hidden_states[layer][0, start:stop] for layer in selected_layers]
    )
    return {
        "states": states.float().cpu().numpy(),
        "layers": np.asarray(selected_layers, dtype=np.int32),
        "token_ids": generated_ids.cpu().numpy().astype(np.int64, copy=False),
        "prompt_length": np.asarray(len(prompt_ids), dtype=np.int32),
    }


def save_full_generated_states(path: str | Path, payload: dict[str, np.ndarray]) -> None:
    """Save the extraction payload without Python object arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
