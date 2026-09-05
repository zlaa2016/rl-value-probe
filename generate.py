import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def prompt_ids_from_row(row, tokenizer):
    """
    Prefer the dataset's exact tokenized RL prompt.
    Fall back to tokenizing the prompt string.
    """
    ids = row.get("input_ids_prompt")
    if ids:
        return torch.tensor(ids, dtype=torch.long)

    text = row["prompt"]
    return tokenizer(text, return_tensors="pt")["input_ids"][0]


@torch.inference_mode()
def generate_rollout(
    model,
    tokenizer,
    prompt_ids,
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.95,
):
    input_ids = prompt_ids.unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output[0, input_ids.shape[1]:].detach().cpu()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "generated_ids": generated_ids,
        "text": text,
    }
