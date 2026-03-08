"""
train_grpo.py — GRPO fine-tuning of Qwen 2.5 3B against the APEX OpenENV environment.

Directory Structure:
    models/
        base_model/          # Base models downloaded from HuggingFace
            Qwen2.5-3B-Instruct/
        fine_tuned/          # Fine-tuned models (auto-versioned)
            Qwen2.5-3B-Instruct_v0/
            Qwen2.5-3B-Instruct_v1/

Setup:
    pip install "trl[openenv]" groq accelerate python-dotenv

    # Download base model first
    python training/download_model.py

    # Terminal 1 — start APEX env server
    cd apex_env && uvicorn server.app:app --host 0.0.0.0 --port 8001

    # Terminal 2 — run training
    # Configure .env: MODEL_NAME=Qwen/Qwen2.5-3B-Instruct (for base model)
    #            or: MODEL_NAME=Qwen2.5-3B-Instruct_v0 (for fine-tuned model)
    python training/train_grpo.py
"""

import os
import sys
from pathlib import Path
from datasets import Dataset
from dotenv import load_dotenv
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

# Load environment variables from .env file in parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

sys.path.insert(0, str(Path(__file__).parent.parent))
from apex_env.client import APEXClient

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — Load from .env file
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME      = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
MODELS_FOLDER   = os.getenv("MODELS_FOLDER", "./models")

# Determine model path based on MODEL_NAME format:
# - "Qwen/Qwen2.5-3B-Instruct" → load from models/base_model/Qwen2.5-3B-Instruct
# - "Qwen2.5-3B-Instruct_v0" → load from models/fine_tuned/Qwen2.5-3B-Instruct_v0
if "/" in MODEL_NAME:
    # HuggingFace ID format - load from base_model
    MODEL_SHORT_NAME = MODEL_NAME.split("/")[-1]
    MODEL_PATH = str(Path(MODELS_FOLDER) / "base_model" / MODEL_SHORT_NAME)
else:
    # Local model name - check fine_tuned first, then base_model
    if (Path(MODELS_FOLDER) / "fine_tuned" / MODEL_NAME).exists():
        MODEL_SHORT_NAME = MODEL_NAME
        MODEL_PATH = str(Path(MODELS_FOLDER) / "fine_tuned" / MODEL_NAME)
    else:
        MODEL_SHORT_NAME = MODEL_NAME
        MODEL_PATH = str(Path(MODELS_FOLDER) / "base_model" / MODEL_NAME)

ENV_URL         = os.getenv("ENV_URL", "http://localhost:8001")
NUM_SCENARIOS   = int(os.getenv("NUM_SCENARIOS", "22"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "8"))
MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS", "512"))
NUM_EPOCHS      = int(os.getenv("NUM_EPOCHS", "3"))

# Auto-version: find next available version number in fine_tuned folder
def get_next_model_version(base_path: Path, model_name: str) -> str:
    """Find the next available version number for saving fine-tuned models."""
    fine_tuned_path = base_path / "fine_tuned"
    fine_tuned_path.mkdir(parents=True, exist_ok=True)

    version = 0
    while True:
        versioned_path = fine_tuned_path / f"{model_name}_v{version}"
        if not versioned_path.exists():
            return str(versioned_path)
        version += 1

OUTPUT_DIR = get_next_model_version(Path(MODELS_FOLDER), MODEL_SHORT_NAME)

# Training hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-5"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.05"))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "5"))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "20"))
SEED = int(os.getenv("SEED", "42"))

# VLLM & Mixed Precision
USE_VLLM = os.getenv("USE_VLLM", "true").lower() == "true"
VLLM_MODE = os.getenv("VLLM_MODE", "colocate")
BF16 = os.getenv("BF16", "true").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# DATASET — dummy counter, just controls how many times rollout_func is called
# ─────────────────────────────────────────────────────────────────────────────
dataset = Dataset.from_list(
    [{"prompt": f"Scenario {i}"} for i in range(NUM_SCENARIOS * NUM_EPOCHS)]
)

# ─────────────────────────────────────────────────────────────────────────────
# ROLLOUT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def rollout_func(prompts: list, trainer: GRPOTrainer) -> dict:
    tokenizer = trainer.processing_class

    with APEXClient(base_url=ENV_URL) as client:
        # env gives us the real scenario prompt
        obs = client.reset()
        real_prompt = obs["prompt"]

        # generate NUM_GENERATIONS completions for this prompt
        outputs = generate_rollout_completions(
            trainer,
            [real_prompt] * NUM_GENERATIONS,
            #max_new_tokens=MAX_NEW_TOKENS,
        )

        # score each completion through the env
        rewards = []
        for out in outputs:
            text = tokenizer.decode(out["completion_ids"], skip_special_tokens=True)
            result = client.step(text)
            reward = (
                float(result.get("reward", 0.0))
                if isinstance(result, dict)
                else float(getattr(result, "reward", 0.0))
            )
            rewards.append(reward)

    return {
        "prompt_ids":     [o["prompt_ids"]     for o in outputs],
        "completion_ids": [o["completion_ids"] for o in outputs],
        "logprobs": [[(lp, tid) for lp, tid in zip(o["logprobs"], o["completion_ids"])] for o in outputs],
        "env_reward":     rewards,
    }

# ─────────────────────────────────────────────────────────────────────────────
# REWARD FUNCTION — rewards already computed above, just pass through
# ─────────────────────────────────────────────────────────────────────────────
def reward_fn(completions, env_reward=None, **kwargs):
    return env_reward if env_reward is not None else [0.0] * len(completions)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=MODEL_PATH,
    reward_funcs=reward_fn,
    rollout_func=rollout_func,
    args=GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=1,         # epochs baked into dataset length
        warmup_steps=int(0.05 * len(dataset)),
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        use_vllm=USE_VLLM,
        vllm_mode=VLLM_MODE,
        seed=SEED,
        max_completion_length=MAX_NEW_TOKENS,
        report_to="none",
        gradient_checkpointing=True
    ),
    train_dataset=dataset,
)

if __name__ == "__main__":
    print(f"🚀  Training model from: {MODEL_PATH}")
    print(f"    Environment: {ENV_URL}")
    print(f"    {NUM_SCENARIOS} scenarios × {NUM_EPOCHS} epochs × {NUM_GENERATIONS} generations/prompt")
    print(f"    Will save to: {OUTPUT_DIR}")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\n✅  Done. Trained model saved to: {OUTPUT_DIR}")
    print(f"\n💡 To continue training from this fine-tuned model:")
    print(f"    Update .env: MODEL_NAME={Path(OUTPUT_DIR).name}")
    # trainer.push_to_hub("your-username/apex-qwen2.5-3b")
