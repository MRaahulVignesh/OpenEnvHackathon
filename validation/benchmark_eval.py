#!/usr/bin/env python3
"""
Evaluation script that gives out the benchmark for the desired model on the Apex Dataset
"""

import os
import sys
import gc
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

EVAL_MODEL    = os.getenv("EVAL_MODEL", os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"))
JUDGE_MODEL   = os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-14B-Instruct")
MODELS_FOLDER = os.getenv("MODELS_FOLDER", "./models")
DATA_DIR      = os.getenv("DATA_DIR", "apex_env/data")


# ============================================================
# MODEL PATH RESOLUTION
# ============================================================
def resolve_model_path(model_name: str, models_folder: str) -> str:
    """
    Find model on disk. Checks fine_tuned/ then base_model/.
    Falls back to HuggingFace ID if HF-style name and not found locally.
    Errors out if local-style name (no '/') and not found on disk.
    """
    base = Path(models_folder)

    if "/" in model_name:
        # HuggingFace ID e.g. "Qwen/Qwen2.5-1.5B-Instruct"
        short_name = model_name.split("/")[-1]
        local = base / "base_model" / short_name
        if (local / "config.json").exists():
            print(f"  ✓ Found locally: {local}")
            return str(local)
        print(f"  ⚠ Not found at {local}, will use HuggingFace ID: {model_name}")
        return model_name
    else:
        # Local name e.g. "Qwen2.5-1.5B-Instruct_v14"
        for subfolder in ["fine_tuned", "base_model"]:
            local = base / subfolder / model_name
            if (local / "config.json").exists():
                print(f"  ✓ Found locally: {local}")
                return str(local)
        raise FileNotFoundError(
            f"Model '{model_name}' not found in:\n"
            f"  {base}/fine_tuned/{model_name}\n"
            f"  {base}/base_model/{model_name}\n"
            f"Run: python training/download_model.py"
        )


# ============================================================
# GPU CLEANUP
# ============================================================
print("=" * 60)
print("GPU CLEANUP")
print("=" * 60)

try:
    del smaller_model, bigger_model, smaller_tokenizer, bigger_tokenizer
except:
    pass

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free_mem  = torch.cuda.mem_get_info()[0] / 1024**3
    total_mem = torch.cuda.mem_get_info()[1] / 1024**3
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"Free memory:  {free_mem:.1f} GB / {total_mem:.1f} GB")
else:
    print("WARNING: No CUDA GPU available!")
print()


# ============================================================
# CONFIGURATION
# ============================================================
print("=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"  EVAL_MODEL    : {EVAL_MODEL}")
print(f"  JUDGE_MODEL   : {JUDGE_MODEL}")
print(f"  MODELS_FOLDER : {MODELS_FOLDER}")
print(f"  DATA_DIR      : {DATA_DIR}")
print()

print("Resolving eval model path...")
SMALL_MODEL_PATH = resolve_model_path(EVAL_MODEL, MODELS_FOLDER)
print()

print("Resolving judge model path...")
BIGGER_MODEL_PATH = resolve_model_path(JUDGE_MODEL, MODELS_FOLDER)
print()


# ============================================================
# LOAD SCENARIOS
# ============================================================
def load_all_scenarios(data_dir: str = DATA_DIR):
    all_scenarios = []
    for category in ["banking", "consulting", "law"]:
        filepath = Path(data_dir) / category / "scenarios.json"
        try:
            with open(filepath) as f:
                scenarios = json.load(f)
            for s in scenarios:
                s["category"] = category
            all_scenarios.extend(scenarios)
            print(f"  Loaded {len(scenarios):2d} {category} scenarios")
        except FileNotFoundError:
            print(f"  {filepath} not found — skipping")
    print(f"  Total: {len(all_scenarios)} scenarios")
    return all_scenarios


def format_prompt(scenario):
    workspace = "=== WORKSPACE FILES ===\n\n"
    for filename, content in scenario["files"].items():
        workspace += f"--- {filename} ---\n{content}\n\n"
    return (
        f"{workspace}\n"
        f"=== YOUR TASK ===\n"
        f"{scenario['task']}\n\n"
        f"Review all files carefully. Produce a professional, complete response.\n"
    )


print("=" * 60)
print("LOADING SCENARIOS")
print("=" * 60)
scenarios = load_all_scenarios()
print()


# ============================================================
# LOAD EVAL MODEL (smaller)
# ============================================================
print("=" * 60)
print(f"LOADING EVAL MODEL")
print("=" * 60)
print(f"  Path: {SMALL_MODEL_PATH}")

smaller_tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_PATH)
smaller_model = AutoModelForCausalLM.from_pretrained(
    SMALL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
smaller_model.eval()
print(f"  ✓ Loaded — {sum(p.numel() for p in smaller_model.parameters())/1e9:.1f}B params")
print()


# ============================================================
# LOAD JUDGE MODEL (bigger)
# ============================================================
print("=" * 60)
print(f"LOADING JUDGE MODEL")
print("=" * 60)
print(f"  Path: {BIGGER_MODEL_PATH}")

bigger_tokenizer = AutoTokenizer.from_pretrained(BIGGER_MODEL_PATH)
bigger_model = AutoModelForCausalLM.from_pretrained(
    BIGGER_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
bigger_model.eval()
print(f"  ✓ Loaded — {sum(p.numel() for p in bigger_model.parameters())/1e9:.1f}B params")
print()


# ============================================================
# INFERENCE FUNCTIONS
# ============================================================
def run_inference_for_smaller_model(prompt: str, max_new_tokens: int = 1024) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a highly skilled professional analyst. "
                "Review the provided workspace files carefully and "
                "respond with accurate, professional analysis."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    text = smaller_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = smaller_tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = smaller_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=smaller_tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return smaller_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_inference_for_bigger_model(prompt: str, max_new_tokens: int = 512) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator of professional analysis responses.",
        },
        {"role": "user", "content": prompt},
    ]
    text = bigger_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = bigger_tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = bigger_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=bigger_tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return bigger_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# SANITY CHECKS
# ============================================================
print("=" * 60)
print("SANITY CHECKS")
print("=" * 60)
print("  Eval model:  ", run_inference_for_smaller_model("What is 2+2? Answer in one word."))
print("  Judge model: ", run_inference_for_bigger_model("What is 2+2? Answer in one word."))
print()


# ============================================================
# JUDGE FUNCTION
# ============================================================
def score_response(scenario: dict, response: str) -> dict:
    rubric_text = "\n".join(
        f"{i+1}. {c}" for i, c in enumerate(scenario["rubric"])
    )
    n = len(scenario["rubric"])

    judge_prompt = f"""You are an expert evaluator assessing a professional analysis response.

TASK GIVEN TO THE MODEL:
{scenario['task']}

MODEL RESPONSE:
{response}

RUBRIC — score each criterion as 1 (met) or 0 (not met):
{rubric_text}

Be strict. Each criterion is binary — no partial credit.
Respond ONLY with valid JSON, no other text:
{{"scores": [list of {n} integers, each 0 or 1], "reasoning": "one sentence"}}"""

    try:
        raw = run_inference_for_bigger_model(judge_prompt, max_new_tokens=256).strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        result = json.loads(raw)
        scores = result["scores"][:n]
        return {
            "criteria_scores": scores,
            "criteria_met":    sum(scores),
            "criteria_total":  n,
            "total_score":     sum(scores) / n,
            "reasoning":       result.get("reasoning", ""),
        }
    except Exception as e:
        print(f"  Judge error: {e}")
        return {
            "criteria_scores": [0] * n,
            "criteria_met":    0,
            "criteria_total":  n,
            "total_score":     0.0,
            "reasoning":       f"Error: {e}",
        }


# ============================================================
# EVALUATION LOOP
# ============================================================
def run_eval(scenarios, max_scenarios=None):
    results = []
    eval_set = scenarios[:max_scenarios] if max_scenarios else scenarios
    total = len(eval_set)

    print(f"  Evaluating {total} scenarios")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 55)

    for i, scenario in enumerate(eval_set):
        print(f"[{i+1}/{total}] {scenario['id']} ({scenario['category']}, {scenario.get('difficulty', 'N/A')})")

        prompt   = format_prompt(scenario)
        t0       = time.time()
        response = run_inference_for_smaller_model(prompt)
        elapsed  = time.time() - t0
        scored   = score_response(scenario, response)

        results.append({
            "id":         scenario["id"],
            "category":   scenario["category"],
            "difficulty": scenario.get("difficulty", "N/A"),
            "world":      scenario.get("world", "N/A"),
            "response":   response,
            "elapsed_s":  round(elapsed, 1),
            **scored,
        })

        pct = scored["total_score"] * 100
        print(f"  Score: {scored['criteria_met']}/{scored['criteria_total']} ({pct:.0f}%)  [{elapsed:.1f}s]")
        print(f"  {scored['reasoning'][:90]}")
        print()
        time.sleep(0.3)

    return results


print("=" * 60)
print("RUNNING EVALUATION")
print("=" * 60)
results = run_eval(scenarios)
print()


# ============================================================
# ANALYSIS
# ============================================================
def analyze_results(results, label="Results"):
    n     = len(results)
    avg   = sum(r["total_score"] for r in results) / n
    pass1 = sum(1 for r in results if r["total_score"] == 1.0) / n

    print(f'\n{"=" * 55}')
    print(f"  {label} — {EVAL_MODEL}")
    print(f'{"=" * 55}')
    print(f"  Scenarios:     {n}")
    print(f"  Avg score:     {avg*100:.1f}%")
    print(f"  Pass@1 (100%): {pass1*100:.1f}%")
    print()

    for cat in ["banking", "consulting", "law"]:
        sub = [r for r in results if r["category"] == cat]
        if sub:
            cat_avg = sum(r["total_score"] for r in sub) / len(sub)
            print(f"  {cat.upper():12s}  {cat_avg*100:.1f}%  ({len(sub)} scenarios)")

    print()
    worst = sorted(results, key=lambda r: r["total_score"])[:3]
    print("  Lowest scores:")
    for r in worst:
        print(f"    {r['id']:15s} {r['total_score']*100:.0f}%  {r['world']}")

    return {"avg_score": avg, "pass_at_1": pass1, "n": n}


print("=" * 60)
print("ANALYSIS")
print("=" * 60)
summary = analyze_results(results, "EVAL")
print()


# ============================================================
# SAVE RESULTS
# ============================================================
# Use eval model name in filename so baseline and post-training don't overwrite each other
safe_name = EVAL_MODEL.replace("/", "_")
output_file = Path(__file__).parent.parent / f"eval_results_{safe_name}.json"

with open(output_file, "w") as f:
    json.dump({
        "eval_model":   EVAL_MODEL,
        "model_path":   SMALL_MODEL_PATH,
        "judge_model":  JUDGE_MODEL,
        "judge_path":   BIGGER_MODEL_PATH,
        "timestamp":    datetime.now().isoformat(),
        "summary":      summary,
        "results":      results,
    }, f, indent=2)

print(f"✓ Saved to {output_file}")
print()


# ============================================================
# WORST RESPONSE INSPECTION
# ============================================================
print("=" * 60)
print("WORST PERFORMING SCENARIO")
print("=" * 60)

worst = min(results, key=lambda r: r["total_score"])
print(f"  Scenario: {worst['id']} — {worst['world']}")
print(f"  Score:    {worst['criteria_met']}/{worst['criteria_total']} ({worst['total_score']*100:.0f}%)")
print()
print("MODEL RESPONSE:")
print("-" * 50)
print(worst["response"])
print()
print("RUBRIC BREAKDOWN:")
scenario = next(s for s in scenarios if s["id"] == worst["id"])
for criterion, score in zip(scenario["rubric"], worst["criteria_scores"]):
    icon = "✓" if score == 1 else "✗"
    print(f"  [{icon}] {criterion}")
print()
print(f"Judge: {worst['reasoning']}")
print()


