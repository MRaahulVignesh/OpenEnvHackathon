"""
train_grpo.py — GRPO fine-tuning of Qwen 2.5 against the APEX OpenENV environment.

Key design: reward functions call the env directly (no rollout_func).
TRL handles generation internally; reward_fn receives completions and
hits the APEX server to score them. This is the pattern that actually works.

Directory Structure:
    models/
        base_model/          # Base models downloaded from HuggingFace
            Qwen2.5-3B-Instruct/
        fine_tuned/          # Fine-tuned models (auto-versioned)
            Qwen2.5-3B-Instruct_v0/

Setup:
    # Terminal 1 — start APEX env server
    cd apex_env && uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Terminal 2 — run training
    python training/train_grpo.py
"""

import os
import sys
from pathlib import Path
from datasets import Dataset
from dotenv import load_dotenv
from trl import GRPOConfig, GRPOTrainer

# Load environment variables from .env file in parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

sys.path.insert(0, str(Path(__file__).parent.parent))
from apex_env.client import APEXClient

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — Load from .env file
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME      = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
MODELS_FOLDER   = os.getenv("MODELS_FOLDER", "./models")

# Determine model path:
# - "Qwen/Qwen2.5-3B-Instruct" → load from models/base_model/Qwen2.5-3B-Instruct
# - "Qwen2.5-3B-Instruct_v0"   → load from models/fine_tuned/Qwen2.5-3B-Instruct_v0
if "/" in MODEL_NAME:
    MODEL_SHORT_NAME = MODEL_NAME.split("/")[-1]
    MODEL_PATH = str(Path(MODELS_FOLDER) / "base_model" / MODEL_SHORT_NAME)
else:
    if (Path(MODELS_FOLDER) / "fine_tuned" / MODEL_NAME).exists():
        MODEL_SHORT_NAME = MODEL_NAME
        MODEL_PATH = str(Path(MODELS_FOLDER) / "fine_tuned" / MODEL_NAME)
    else:
        MODEL_SHORT_NAME = MODEL_NAME
        MODEL_PATH = str(Path(MODELS_FOLDER) / "base_model" / MODEL_NAME)

ENV_URL         = os.getenv("ENV_URL", "http://localhost:8000")
NUM_SCENARIOS   = int(os.getenv("NUM_SCENARIOS", "22"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "4"))
MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS", "2048"))
NUM_EPOCHS      = int(os.getenv("NUM_EPOCHS", "3"))

# Auto-version output dir
def get_next_model_version(base_path: Path, model_name: str) -> str:
    fine_tuned_path = base_path / "fine_tuned"
    fine_tuned_path.mkdir(parents=True, exist_ok=True)
    version = 0
    while True:
        versioned_path = fine_tuned_path / f"{model_name}_v{version}"
        if not versioned_path.exists():
            return str(versioned_path)
        version += 1

OUTPUT_DIR = get_next_model_version(Path(MODELS_FOLDER), MODEL_SHORT_NAME)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Training hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8"))
LEARNING_RATE  = float(os.getenv("LEARNING_RATE", "1e-5"))
LOGGING_STEPS  = int(os.getenv("LOGGING_STEPS", "1"))
SAVE_STEPS     = int(os.getenv("SAVE_STEPS", "20"))
SEED           = int(os.getenv("SEED", "42"))
BF16           = os.getenv("BF16", "true").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# APEX CLIENT — single global connection, reused across all reward calls
# ─────────────────────────────────────────────────────────────────────────────
apex_client = APEXClient(base_url=ENV_URL)
apex_client.__enter__()

# ─────────────────────────────────────────────────────────────────────────────
# DATASET — chat-formatted prompts (TRL requires this format)
# The placeholder content is replaced by the real scenario prompt inside
# reward_fn via client.reset(). TRL just needs a valid chat structure.
# ─────────────────────────────────────────────────────────────────────────────
dataset = Dataset.from_list([
    {
        "prompt": [
            {"role": "user", "content": f"Scenario {i % NUM_SCENARIOS}"}
        ]
    }
    for i in range(NUM_SCENARIOS * NUM_EPOCHS)
])

# ─────────────────────────────────────────────────────────────────────────────
# REWARD FUNCTION
# TRL generates completions internally and passes them here.
# We hit the APEX env server to score each one.
#
# Why this works (unlike rollout_func):
#   - TRL calls reward_fn directly after its own generation loop
#   - No experimental API wiring issues
#   - Matches the proven pattern from the working example notebook
# ─────────────────────────────────────────────────────────────────────────────
def reward_fn(completions, **kwargs):
    rewards = []
    for completion in completions:
        # Completions arrive as chat messages: [{"role": "assistant", "content": "..."}]
        if isinstance(completion, list):
            text = completion[0]["content"]
        else:
            text = str(completion)

        try:
            apex_client.reset()
            result = apex_client.step(text)

            if isinstance(result, dict):
                reward = float(result.get("reward", 0.0))
            else:
                reward = float(getattr(result, "reward", 0.0))

        except Exception as e:
            print(f"  ⚠ Env error: {e}")
            reward = 0.0

        print(f"  ✓ reward={reward:.4f}  len={len(text.split())} words")
        rewards.append(reward)

    return rewards

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=MODEL_PATH,
    reward_funcs=reward_fn,
    args=GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_NEW_TOKENS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=1,             # epochs baked into dataset length
        warmup_steps=int(0.05 * NUM_SCENARIOS * NUM_EPOCHS),
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        save_strategy="no",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        seed=SEED,
        report_to="none",
        gradient_checkpointing=True,
    ),
    train_dataset=dataset,
)

if __name__ == "__main__":
    print(f"🚀  Training model: {MODEL_PATH}")
    print(f"    Environment:    {ENV_URL}")
    print(f"    {NUM_SCENARIOS} scenarios × {NUM_EPOCHS} epochs × {NUM_GENERATIONS} generations/prompt")
    print(f"    Max tokens:     {MAX_NEW_TOKENS}")
    print(f"    Will save to:   {OUTPUT_DIR}")
    print()

    try:
        trainer.train()
        trainer.save_model(OUTPUT_DIR)
        print(f"\n✅  Done. Model saved to: {OUTPUT_DIR}")
        print(f"\n💡  To continue from this checkpoint:")
        print(f"    Update .env: MODEL_NAME={Path(OUTPUT_DIR).name}")
    finally:
        apex_client.__exit__(None, None, None)