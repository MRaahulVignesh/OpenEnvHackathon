# APEX Environment

**APEX** (Advanced Professional EXperience) is a reinforcement learning environment for training LLM agents on expert-level professional tasks across investment banking, management consulting, and corporate law. Built on the [OpenENV](https://github.com/openenv) framework for the OpenENV Hackathon SF 2026.

---

## What It Is

APEX exposes a standard OpenENV interface (`reset()` / `step()`) backed by 22 hand-crafted professional scenarios. Each scenario gives the agent a workspace of real-looking files — financial models, legal agreements, client emails — and a task. The agent produces a professional response. The environment scores it.

The environment is not a static benchmark. It adapts to the agent: difficulty escalates as performance improves, scenarios are adversarially corrupted on hard episodes, and every response is scored relative to a gold reference output — not just against a fixed checklist.

---

## Scenarios

22 scenarios across three domains, organised by difficulty tier:

| Domain | Easy | Medium | Hard | Total |
|---|---|---|---|---|
| Investment Banking | 2 | 2 | 4 | 8 |
| Management Consulting | 2 | 2 | 3 | 7 |
| Corporate Law | 2 | 3 | 2 | 7 |
| **Total** | **6** | **9** | **7** | **22** |

Each scenario contains:
- **`files`** — 3–4 workspace files (financials, memos, emails, policy excerpts)
- **`task`** — the specific instruction to the agent
- **`rubric`** — 3–7 binary criteria the judge evaluates against
- **`gold_output`** — a reference expert response used for peer comparison scoring
- **`difficulty`** — `easy` | `medium` | `hard`

---

## Environment Features

### 1. LLM-as-Judge Scoring

Responses are scored by a local judge model (`Qwen2.5-7B-Instruct`, loaded via `transformers` in `bfloat16`). The judge receives the task, the agent's response, and the rubric, and returns a binary score per criterion plus a one-sentence reasoning trace.

```
score = criteria_met / criteria_total   →  base_reward ∈ [0.0, 1.0]
```

The judge uses greedy decoding (`do_sample=False`) for deterministic, reproducible scores. It runs as a separate process from the training vLLM instance to avoid CUDA allocator conflicts.

---

### 2. Peer Comparison Reward

Instead of scoring only against an absolute rubric, each response is also scored **relative to the scenario's gold output**. This gives the agent a richer gradient signal: it is rewarded for beating the reference, not just for passing checkboxes.

**How it works:**

1. Judge scores the agent's response → `base_reward`
2. Judge scores `gold_output` against the same rubric → `gold_score` (cached per scenario — scored once for the entire training run, never again)
3. Relative reward is computed:

```
delta          = base_reward - gold_score          # ∈ [-1.0, 1.0]
peer_reward    = 0.5 + (delta / 2)                 # 0.5 = matches gold, >0.5 = beats gold
blended_reward = 0.6 × peer_reward + 0.4 × base_reward
```

The 60/40 blend keeps an absolute signal so the agent cannot exploit a weak gold baseline.

---

### 3. Difficulty Progression

The environment tracks a **rolling average reward over the last 5 episodes per domain** and adjusts the difficulty tier it selects scenarios from:

| Condition | Action |
|---|---|
| Rolling avg ≥ 0.75 | Escalate to next tier (easy → medium → hard) |
| Rolling avg ≤ 0.40 | De-escalate to previous tier |
| Otherwise | Stay at current tier |

Each domain (`banking`, `consulting`, `law`) progresses independently. `reset()` picks a random scenario from the current tier for that domain, with fallback to adjacent tiers if a tier has no scenarios.

This means an agent that has mastered easy banking tasks will stop seeing them — the environment pushes it toward harder material automatically.

---

### 4. Adversarial Noise Injection

On **hard scenarios**, the environment optionally corrupts one file with a plausible but incorrect figure before presenting the observation. The agent is rewarded for detecting and flagging the discrepancy.

**Example (bank_003):**
```
Original:  EBITDA: $4.1M (FY2025)
Injected:  EBITDA: $8.2M (FY2025)   ← planted error
```

Detection is keyword-based: if the agent's response contains terms like `"EBITDA discrepancy"`, `"conflicting EBITDA"`, etc., it is credited.

```
noise_bonus = +0.2  if injected error was flagged
            =  0.0  otherwise
```

The final reward combines all signals:

```
final_reward = min(1.0, blended_reward + noise_bonus)
```

Noise injection is a deep copy operation — the original scenario is never mutated.

---

## Reward Breakdown

Every `step()` returns a full reward breakdown in `APEXObservation`:

| Field | Description |
|---|---|
| `base_reward` | Raw rubric score: `criteria_met / criteria_total` |
| `gold_score` | Judge score of the gold reference response |
| `peer_reward` | Relative reward: 0.5 = matches gold, >0.5 = beats gold |
| `blended_reward` | 60% peer + 40% absolute, before noise bonus |
| `noise_bonus` | +0.2 if injected noise was detected, else 0.0 |
| `reward` | Final scalar passed to the training loop |
| `noise_injected` | Whether this episode had noise injected |
| `noise_detected` | Whether the agent flagged it |
| `difficulty` | Scenario difficulty tier for this episode |
| `tier_status` | Current difficulty tier per domain |

---

## File Structure

```
apex_env/
├── server/
│   ├── app.py                  # FastAPI app via openenv.core create_app()
│   ├── apex_environment.py     # Main environment: difficulty, noise, peer reward
│   ├── scorer.py               # RLScorer wrapper around LLMJudge
│   └── llm_judge.py            # Transformers-based judge with gold score cache
├── data/
│   ├── banking/scenarios.json
│   ├── consulting/scenarios.json
│   └── law/scenarios.json
├── client.py                   # APEXClient wrapping GenericEnvClient
└── models.py                   # APEXAction, APEXObservation (Pydantic)

training/
├── train_grpo.py               # GRPO training loop (TRL 0.29)
└── baseline_eval.py            # Pre/post training evaluation
```

---

## Running Locally

**Start the environment server:**
```bash
source .venv/bin/activate
cd apex_env && uvicorn server.app:app --host 0.0.0.0 --port 8000
```

**Run training:**
```bash
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python training/train_grpo.py
```

**Run evaluation:**
```bash
# Baseline (pre-training)
EVAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct python training/baseline_eval.py

# Post-training
EVAL_MODEL=Qwen2.5-1.5B-Instruct_v14 python training/baseline_eval.py
```

---

## Key `.env` Parameters

```bash
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct
MODELS_FOLDER=./models
DATA_DIR=apex_env/data
ENV_URL=http://localhost:8000
NUM_EPOCHS=3
NUM_GENERATIONS=4
```

---

## Training

GRPO (Group Relative Policy Optimization) via TRL 0.29. The training loop uses `reward_fn` — TRL generates completions internally and passes them to `reward_fn`, which calls `client.reset(scenario_id=...)` and `client.step(response)` to get the final scalar reward. Dataset rows include `scenario_id` so the model is always scored against the exact scenario it was shown.

vLLM runs in `colocate` mode alongside the judge model. Memory is managed via `vllm_gpu_memory_utilization=0.2` and `optim=adamw_torch` (no bitsandbytes dependency).

--