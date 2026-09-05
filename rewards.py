import ast
import json
from pathlib import Path
import sys


def _load_instruction_registry():
    """Load only the lightweight IFEvalG registry from Open-Instruct source."""
    source_root = Path(__file__).resolve().parent / "external" / "open-instruct"
    if not source_root.exists():
        raise ImportError(
            "Open-Instruct source is required for exact IF rewards. "
            "Clone it to external/open-instruct; do not install the full package."
        )

    source_path = str(source_root)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

    try:
        from open_instruct.IFEvalG import instructions_registry
    except ImportError as exc:
        raise ImportError(
            "Could not import the lightweight IFEval dependencies. "
            "Run `pip install -r requirements.txt` and download NLTK punkt data."
        ) from exc

    return instructions_registry


def _ifeval_label(row):
    """
    Ai2's IFEvalVerifier expects a serialized list-like constraint label.
    Dolci-Think-RL-7B stores ground_truth as a one-element list.
    """
    return str(row["ground_truth"])


def _remove_thinking_section(prediction):
    """Match Open-Instruct's answer cleanup before constraint checking."""
    prediction = prediction.replace("<|assistant|>", "").strip()
    prediction = prediction.split("</think>")[-1]
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


def ifeval_reward(row, generated_text, generated_ids):
    """
    Compute the same style of terminal instruction-following reward used
    by Ai2's IFEval verifier: fraction of constraints satisfied.
    """
    del generated_ids  # The IFEval constraint checks operate on decoded text.

    instructions_registry = _load_instruction_registry()
    constraint_dict = ast.literal_eval(_ifeval_label(row))[0]
    if isinstance(constraint_dict, str):
        constraint_dict = json.loads(constraint_dict)

    answer = _remove_thinking_section(generated_text)
    if not generated_text or not answer:
        return 0.0

    rewards = []
    instruction_dict = instructions_registry.INSTRUCTION_DICT

    for instruction_key, args in zip(
        constraint_dict["instruction_id"],
        constraint_dict["kwargs"],
    ):
        args = {} if args is None else args
        args = {key: value for key, value in args.items() if value is not None}

        instruction_cls = instruction_dict[instruction_key]
        instruction = instruction_cls(instruction_key)
        instruction.build_description(**args)
        rewards.append(float(instruction.check_following(answer)))

    return float(sum(rewards) / max(len(rewards), 1))
