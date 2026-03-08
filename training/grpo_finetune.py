"""
train_grpo.py — GRPO fine-tuning against the APEX OpenENV environment.
"""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
load_dotenv()

from apex_env.client import APEXClient


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL = os.getenv("BASE_MODEL")
MODELS_FOLDER = os.getenv("MODELS_FOLDER")
ENV_URL = os.getenv("ENV_URL")
DATA_DIR = os.getenv("DATA_DIR")

SEED = int(os.getenv("SEED", "42"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS"))

PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
USE_VLLM = os.getenv("USE_VLLM", "true").lower() == "true"
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS"))
BF16 = os.getenv("BF16", "true").lower() == "true"


# ─────────────────────────────────────────────
# MODEL PATH RESOLUTION
# ─────────────────────────────────────────────
def resolve_model_path(model_name: str) -> tuple[str, str]:
    if "/" in model_name:
        short = model_name.split("/")[-1]
        return short, str(Path(MODELS_FOLDER) / "base_model" / short)

    short = model_name
    ft = Path(MODELS_FOLDER) / "fine_tuned" / short
    base = Path(MODELS_FOLDER) / "base_model" / short
    return short, str(ft if ft.exists() else base)


def next_version(base: Path, name: str) -> str:
    out = base / "fine_tuned"
    out.mkdir(parents=True, exist_ok=True)

    v = 0
    while (out / f"{name}_v{v}").exists():
        v += 1

    return str(out / f"{name}_v{v}")


MODEL_SHORT, MODEL_PATH = resolve_model_path(BASE_MODEL)
OUTPUT_DIR = next_version(Path(MODELS_FOLDER), MODEL_SHORT)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def format_prompt(scenario: dict) -> str:
    workspace = "=== WORKSPACE FILES ===\n\n"

    for name, content in scenario["files"].items():
        workspace += f"--- {name} ---\n{content}\n\n"

    return (
        f"{workspace}"
        f"=== YOUR TASK ===\n"
        f"{scenario['task']}\n\n"
        "Review all files carefully and produce a professional response."
    )


def load_scenarios(data_dir: str) -> list[dict]:
    scenarios = []
    base = Path(data_dir)

    for category in ["banking", "consulting", "law"]:
        path = base / category / "scenarios.json"

        if not path.exists():
            continue

        with open(path) as f:
            for s in json.load(f):
                s["category"] = category
                scenarios.append(s)

    if not scenarios:
        raise FileNotFoundError(f"No scenarios found in {data_dir}")

    return scenarios


scenarios = load_scenarios(DATA_DIR)
print(f"Loaded {len(scenarios)} scenarios")


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
rows = []

for _ in range(NUM_EPOCHS):
    for s in scenarios:
        rows.append(
            {
                "prompt": [{"role": "user", "content": format_prompt(s)}],
                "scenario_id": s["id"],
            }
        )

dataset = Dataset.from_list(rows)

print(f"Dataset size: {len(dataset)}")


# ─────────────────────────────────────────────
# ENV CLIENT
# ─────────────────────────────────────────────
apex_client = APEXClient(base_url=ENV_URL)
apex_client.__enter__()


# ─────────────────────────────────────────────
# REWARD FUNCTION
# ─────────────────────────────────────────────
def reward_fn(completions, scenario_id=None, **kwargs):
    rewards = []

    for i, completion in enumerate(completions):
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )

        sid = scenario_id[i] if isinstance(scenario_id, list) else scenario_id

        try:
            apex_client.reset(scenario_id=sid)
            result = apex_client.step(text)

            if isinstance(result, dict):
                reward = float(result.get("reward", 0.0))
                obs = result
            else:
                reward = float(getattr(result, "reward", 0.0))
                obs = getattr(result, "observation", {}) or {}

            reasoning = obs.get("reasoning", "")
            difficulty = obs.get("difficulty", "?")

        except Exception as e:
            print(f"Env error [{sid}]: {e}")
            reward = 0.0
            reasoning = ""
            difficulty = "?"

        print(f"[{sid}] ({difficulty}) reward={reward:.3f}")
        if reasoning:
            print(f" judge: {reasoning[:80]}")

        rewards.append(reward)

    return rewards


# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────
trainer = GRPOTrainer(
    model=MODEL_PATH,
    reward_funcs=reward_fn,
    train_dataset=dataset,
    args=GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_NEW_TOKENS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=1,
        warmup_steps=int(0.05 * len(dataset)),
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        save_strategy="no",
        use_vllm=USE_VLLM,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.2,
        optim="adamw_torch",
        seed=SEED,
        report_to="none",
        gradient_checkpointing=True,
    ),
)


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print(f"\nTraining model: {MODEL_PATH}")
    print(f"Environment:   {ENV_URL}")
    print(f"Dataset size:  {len(dataset)}")
    print(f"Output dir:    {OUTPUT_DIR}\n")

    try:
        trainer.train()
        trainer.save_model(OUTPUT_DIR)
        print(f"\nModel saved to {OUTPUT_DIR}")
    finally:
        apex_client.__exit__(None, None, None)