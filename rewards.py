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


def _parse_ifeval_constraint(row):
    """Unwrap Dolci's nested list/string serialization into one constraint dict."""
    value = row["ground_truth"]

    # Dataset rows currently look like:
    #   ["[{'instruction_id': [...], 'kwargs': [...]}]"]
    # Repeatedly unwrap the singleton containers and serialized values instead
    # of assuming the inner string is strict JSON.
    for _ in range(6):
        if isinstance(value, dict):
            return value

        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(
                    "Expected exactly one IFEval constraint payload, "
                    f"but found {len(value)}."
                )
            value = value[0]
            continue

        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Could not parse the serialized IFEval constraint payload."
                    ) from exc
            continue

        raise TypeError(
            "Unsupported IFEval constraint payload type: "
            f"{type(value).__name__}."
        )

    raise ValueError("IFEval constraint payload was nested too deeply.")


def _remove_thinking_section(prediction):
    """Match Open-Instruct's answer cleanup before constraint checking."""
    prediction = prediction.replace("<|assistant|>", "").strip()
    prediction = prediction.split("</think>")[-1]
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


def ifeval_reward_details(row, generated_text, generated_ids):
    """
    Return terminal reward plus each verifiable constraint's pass/fail result.
    """
    del generated_ids  # The IFEval constraint checks operate on decoded text.

    instructions_registry = _load_instruction_registry()
    constraint_dict = _parse_ifeval_constraint(row)

    answer = _remove_thinking_section(generated_text)
    rewards = []
    constraint_results = []
    instruction_dict = instructions_registry.INSTRUCTION_DICT

    for instruction_key, args in zip(
        constraint_dict["instruction_id"],
        constraint_dict["kwargs"],
    ):
        args = {} if args is None else args
        args = {key: value for key, value in args.items() if value is not None}

        if not generated_text or not answer:
            passed = False
        else:
            instruction_cls = instruction_dict[instruction_key]
            instruction = instruction_cls(instruction_key)
            instruction.build_description(**args)
            passed = bool(instruction.check_following(answer))
        rewards.append(float(passed))
        constraint_results.append({
            "instruction_id": instruction_key,
            "kwargs": args,
            "passed": passed,
        })

    return {
        "reward": float(sum(rewards) / max(len(rewards), 1)),
        "constraint_results": constraint_results,
    }


def ifeval_reward(row, generated_text, generated_ids):
    """Backward-compatible scalar reward wrapper."""
    return ifeval_reward_details(row, generated_text, generated_ids)["reward"]
