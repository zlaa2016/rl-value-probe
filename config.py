MODELS = {
    "base": "allenai/Olmo-3-1025-7B",
    "sft": "allenai/Olmo-3-7B-Think-SFT",
    "dpo": "allenai/Olmo-3-7B-Think-DPO",
    "rlvr": "allenai/Olmo-3-7B-Think",
}

DATASET = "allenai/Dolci-Think-RL-7B"
DATASET_REVISION = "0fb6466d31ef3a9dd16985ef635e6429e05a6491"
IF_ORIGINAL_DATASET = "hamishivi/IF_multi_constraints_upto5_filtered"

DEFAULT_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95

WANDB_ENTITY = "zl6113-new-york-university"
WANDB_PROJECT = "rl-interp"
