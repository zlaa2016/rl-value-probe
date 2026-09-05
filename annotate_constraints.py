import argparse
import json
from pathlib import Path

from data import load_if_prompts
from rewards import ifeval_reward_details


def annotate_rollouts(records, dataset_rows):
    dataset_by_id = {
        str(row.get("custom_id", index)): row
        for index, row in enumerate(dataset_rows)
    }
    annotated = []
    for record in records:
        prompt_id = str(record["prompt_id"])
        if prompt_id not in dataset_by_id:
            raise RuntimeError(f"Could not recover dataset row for {prompt_id}")
        dataset_row = dataset_by_id[prompt_id]
        details = ifeval_reward_details(
            row=dataset_row,
            generated_text=record["generated_text"],
            generated_ids=record.get("generated_token_ids"),
        )
        updated = dict(record)
        updated["reward"] = details["reward"]
        updated["constraint_results"] = details["constraint_results"]
        updated["prompt_text"] = str(dataset_row.get("prompt", ""))
        annotated.append(updated)
    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.rollouts) as f:
        records = [json.loads(line) for line in f if line.strip()]
    prompt_count = len({str(record["prompt_id"]) for record in records})
    annotated = annotate_rollouts(records, load_if_prompts(prompt_count))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with open(temporary, "w") as f:
        for record in annotated:
            f.write(json.dumps(record) + "\n")
    temporary.replace(output_path)
    print(f"Annotated {len(annotated)} rollouts -> {output_path}")


if __name__ == "__main__":
    main()
