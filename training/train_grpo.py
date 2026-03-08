"""
train_grpo.py — GRPO fine-tuning against the APEX OpenENV environment.

Flow:
  1. Load all scenario JSONs from disk at startup
  2. Build dataset with REAL prompts (workspace files + task) + scenario_id
  3. TRL generates completions using real prompts
  4. reward_fn receives completion + scenario_id (from dataset row)
  5. client.reset(scenario_id=...) pins env to the correct scenario
  6. client.step(completion) scores against the matched rubric

Setup:
    # Terminal 1 — start APEX env server
    cd apex_env && uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Terminal 2 — run training
    python training/train_grpo.py
"""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datasets import Dataset
from dotenv import load_dotenv
from trl import GRPOConfig, GRPOTrainer

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

sys.path.insert(0, str(Path(__file__).parent.parent))
from apex_env.client import APEXClient

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME      = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
MODELS_FOLDER   = os.getenv("MODELS_FOLDER", "./models")
ENV_URL         = os.getenv("ENV_URL", "http://localhost:8000")
DATA_DIR        = os.getenv("DATA_DIR", "apex_env/data")
NUM_EPOCHS      = int(os.getenv("NUM_EPOCHS", "3"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "4"))
MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS", "2048"))

PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8"))
LEARNING_RATE  = float(os.getenv("LEARNING_RATE", "1e-5"))
LOGGING_STEPS  = int(os.getenv("LOGGING_STEPS", "1"))
SEED           = int(os.getenv("SEED", "42"))
BF16           = os.getenv("BF16", "true").lower() == "true"

if "/" in MODEL_NAME:
    MODEL_SHORT_NAME = MODEL_NAME.split("/")[-1]
    MODEL_PATH = str(Path(MODELS_FOLDER) / "base_model" / MODEL_SHORT_NAME)
else:
    MODEL_SHORT_NAME = MODEL_NAME
    ft = Path(MODELS_FOLDER) / "fine_tuned" / MODEL_NAME
    MODEL_PATH = str(ft if ft.exists() else Path(MODELS_FOLDER) / "base_model" / MODEL_NAME)

def get_next_version(base: Path, name: str) -> str:
    (base / "fine_tuned").mkdir(parents=True, exist_ok=True)
    v = 0
    while (base / "fine_tuned" / f"{name}_v{v}").exists():
        v += 1
    return str(base / "fine_tuned" / f"{name}_v{v}")

OUTPUT_DIR = get_next_version(Path(MODELS_FOLDER), MODEL_SHORT_NAME)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD SCENARIOS + BUILD PROMPTS
# Matches _format_prompt() in apex_environment.py exactly
# ─────────────────────────────────────────────────────────────────────────────
def format_prompt(scenario: dict) -> str:
    workspace = "=== WORKSPACE FILES ===\n\n"
    for filename, content in scenario["files"].items():
        workspace += f"--- {filename} ---\n{content}\n\n"
    return (
        f"{workspace}\n"
        f"=== YOUR TASK ===\n"
        f"{scenario['task']}\n\n"
        f"Review all files carefully. "
        f"Produce a professional, complete response for the intended audience.\n"
    )

def load_scenarios(data_dir: str) -> list[dict]:
    scenarios = []
    base = Path(data_dir)
    for category in ["banking", "consulting", "law"]:
        path = base / category / "scenarios.json"
        if path.exists():
            with open(path) as f:
                for s in json.load(f):
                    s["category"] = category
                    scenarios.append(s)
    if not scenarios:
        raise FileNotFoundError(f"No scenarios found in {data_dir}")
    return scenarios

scenarios = load_scenarios(DATA_DIR)
print(f"✓ Loaded {len(scenarios)} scenarios")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# Each row: real prompt (files + task) + scenario_id for reward lookup
# Repeated NUM_EPOCHS times so training sees each scenario multiple times
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for _ in range(NUM_EPOCHS):
    for s in scenarios:
        rows.append({
            "prompt": [{"role": "user", "content": format_prompt(s)}],
            "scenario_id": s["id"],
        })

dataset = Dataset.from_list(rows)
print(f"✓ Dataset: {len(dataset)} rows ({len(scenarios)} scenarios × {NUM_EPOCHS} epochs)")

# ─────────────────────────────────────────────────────────────────────────────
# CLIENT — single global connection
# ─────────────────────────────────────────────────────────────────────────────
apex_client = APEXClient(base_url=ENV_URL)
apex_client.__enter__()

# ─────────────────────────────────────────────────────────────────────────────
# REWARD FUNCTION
#
# TRL passes to reward_fn:
#   completions  — generated responses (num_generations per prompt)
#   scenario_id  — from dataset row, repeated num_generations times by TRL
#
# For each completion:
#   1. client.reset(scenario_id=sid) → env pins _current to that scenario
#   2. client.step(text)             → scored against the correct rubric
# ─────────────────────────────────────────────────────────────────────────────
def reward_fn(completions, scenario_id=None, **kwargs):
    rewards = []

    for i, completion in enumerate(completions):
        # Decode text — TRL passes plain strings in colocate mode
        if isinstance(completion, list):
            text = completion[0]["content"] if completion else ""
        else:
            text = str(completion)

        # scenario_id is repeated num_generations times by TRL
        sid = scenario_id[i] if isinstance(scenario_id, list) else scenario_id

        try:
            obs = apex_client.reset(scenario_id=sid)
            returned_sid = obs.get("scenario_id") if isinstance(obs, dict) else getattr(obs, "scenario_id", None)

            result = apex_client.step(text)

            if isinstance(result, dict):
                reward   = float(result.get("reward", 0.0))
                obs_data = result
            else:
                reward   = float(getattr(result, "reward", 0.0))
                obs_data = getattr(result, "observation", {}) or {}
                if not isinstance(obs_data, dict):
                    obs_data = {}

            reasoning        = obs_data.get("reasoning", "")
            criteria_met     = obs_data.get("criteria_met", "?")
            criteria_total   = obs_data.get("criteria_total", "?")
            criteria         = f"{criteria_met}/{criteria_total}"
            difficulty       = obs_data.get("difficulty", "?")
            base_reward      = obs_data.get("base_reward", "?")
            gold_score       = obs_data.get("gold_score", "?")
            peer_reward      = obs_data.get("peer_reward", "?")
            noise_injected   = obs_data.get("noise_injected", False)
            noise_detected   = obs_data.get("noise_detected", False)
            citation_bonus   = obs_data.get("citation_bonus", 0.0)
            file_citations   = obs_data.get("file_citations", [])
            figure_citations = obs_data.get("figure_citations", [])
            tier_status      = obs_data.get("tier_status", {})

        except Exception as e:
            print(f"  ⚠ Env error [{sid}]: {e}")
            reward = 0.0
            reasoning = criteria = difficulty = ""
            base_reward = gold_score = peer_reward = citation_bonus = 0.0
            noise_injected = noise_detected = False
            file_citations = figure_citations = []
            tier_status = {}

        # ── Rich training log ─────────────────────────────────────────────────
        noise_str    = ""
        if noise_injected:
            noise_str = f"  💉 noise={'✓ detected' if noise_detected else '✗ missed'}"

        citation_str = ""
        if citation_bonus and float(citation_bonus) > 0:
            all_cites = list(file_citations or []) + list(figure_citations or [])
            citation_str = f"  📎 citations={all_cites[:4]} +{float(citation_bonus):.2f}"

        tier_str = ""
        if tier_status:
            tier_str = "  tiers: " + " | ".join(f"{k}={v}" for k, v in tier_status.items())

        print(
            f"\n  ┌─ [{sid}] ({difficulty})"
            f"\n  │  rubric : {criteria} criteria met"
            f"\n  │  scores : base={float(base_reward):.2f}  gold={float(gold_score):.2f}  peer={float(peer_reward):.2f}  citation=+{float(citation_bonus):.2f}"
            f"\n  │  final  : {reward:.4f}"
            + (f"\n  │  noise  : {'injected + detected ✓' if noise_detected else 'injected, missed ✗' if noise_injected else 'none'}" if noise_injected else "")
            + (f"\n  │  cites  : {list(file_citations or []) + list(figure_citations or [])[:4]}" if citation_bonus and float(citation_bonus) > 0 else "")
            + (f"\n  │  judge  : {reasoning[:80]}" if reasoning else "")
            + (f"\n  └─ {tier_str}" if tier_str else "  └─")
        )
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
        num_train_epochs=1,
        warmup_steps=int(0.05 * len(dataset)),
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        save_strategy="no",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.2,
        optim="adamw_torch",
        seed=SEED,
        report_to="none",
        gradient_checkpointing=True,
    ),
    train_dataset=dataset,
)

if __name__ == "__main__":
    print(f"\n🚀  Training: {MODEL_PATH}")
    print(f"    Env:      {ENV_URL}")
    print(f"    Dataset:  {len(dataset)} rows  |  {NUM_GENERATIONS} generations/prompt")
    print(f"    Output:   {OUTPUT_DIR}\n")

    try:
        trainer.train()
        trainer.save_model(OUTPUT_DIR)
        print(f"\n✅  Saved to: {OUTPUT_DIR}")
        print(f"    Next run: MODEL_NAME={Path(OUTPUT_DIR).name}")
    finally:
        apex_client.__exit__(None, None, None)