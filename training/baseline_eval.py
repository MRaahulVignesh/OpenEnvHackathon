#!/usr/bin/env python3
"""
baseline_eval.py — Baseline Evaluation for APEX Environment

Evaluates a base model (or fine-tuned model) against all APEX scenarios using
Qwen 2.5 14B as LLM-as-Judge. Saves results to baseline_results.json for
comparison after training.

Directory Structure:
    models/
        base_model/          # Base models downloaded from HuggingFace
            Qwen2.5-3B-Instruct/
        fine_tuned/          # Fine-tuned models (auto-versioned)
            Qwen2.5-3B-Instruct_v0/

Setup:
    # 1. Create .env file in project root with MODEL_NAME and MODELS_FOLDER
    # 2. Download base model
    python training/download_model.py

    # 3. Run baseline evaluation
    python training/baseline_eval.py

    # To evaluate a fine-tuned model, update MODEL_NAME in .env:
    # MODEL_NAME=Qwen2.5-3B-Instruct_v0

Usage:
    python training/baseline_eval.py
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

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
MODELS_FOLDER = os.getenv("MODELS_FOLDER", "./models")
DATA_DIR = os.getenv("DATA_DIR", "apex_env/data")


# ============================================================
# GPU CLEANUP — Run this at the beginning to free memory
# ============================================================
print("=" * 60)
print("GPU CLEANUP")
print("=" * 60)

# Delete any existing models if they're in memory
try:
    del smaller_model
    del bigger_model
    del smaller_tokenizer
    del bigger_tokenizer
except:
    pass

# Force garbage collection
gc.collect()

# Clear PyTorch cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    total_mem = torch.cuda.mem_get_info()[1] / 1024**3
    print(f"Free memory: {free_mem:.1f} GB")
    print(f"Total memory: {total_mem:.1f} GB")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No CUDA GPU available!")

print()


# ============================================================
# CONFIGURATION
# ============================================================
print("=" * 60)
print("CONFIGURATION")
print("=" * 60)

# Determine model path based on MODEL_NAME format:
# - "Qwen/Qwen2.5-3B-Instruct" → load from models/base_model/Qwen2.5-3B-Instruct
# - "Qwen2.5-3B-Instruct_v0" → load from models/fine_tuned/Qwen2.5-3B-Instruct_v0
if "/" in MODEL_NAME:
    # HuggingFace ID format - load from base_model
    model_short_name = MODEL_NAME.split("/")[-1]
    local_model_path = Path(MODELS_FOLDER) / "base_model" / model_short_name
else:
    # Local model name - check fine_tuned first, then base_model
    model_short_name = MODEL_NAME
    if (Path(MODELS_FOLDER) / "fine_tuned" / MODEL_NAME).exists():
        local_model_path = Path(MODELS_FOLDER) / "fine_tuned" / MODEL_NAME
    else:
        local_model_path = Path(MODELS_FOLDER) / "base_model" / MODEL_NAME

# Check if model exists locally, otherwise use Hugging Face ID
if (local_model_path / "config.json").exists():
    SMALL_MODEL_PATH = str(local_model_path)
    print(f"✓ Found local model: {SMALL_MODEL_PATH}")
else:
    SMALL_MODEL_PATH = MODEL_NAME if "/" in MODEL_NAME else f"Qwen/{MODEL_NAME}"
    print(f"⚠ Local model not found at {local_model_path}")
    print(f"  Falling back to Hugging Face: {SMALL_MODEL_PATH}")
    print(f"  TIP: Run 'python training/download_model.py' first")

print(f"  MODEL_NAME    : {MODEL_NAME}")
print(f"  MODEL_PATH    : {SMALL_MODEL_PATH}")
print(f"  MODELS_FOLDER : {MODELS_FOLDER}")
print(f"  DATA_DIR      : {DATA_DIR}")
print()

print('Using local Qwen 14B model as LLM-as-Judge')
print()


# ============================================================
# LOAD SCENARIOS
# ============================================================
def load_all_scenarios(data_dir: str = DATA_DIR):
    """Load all scenarios from banking, consulting, and law categories."""
    all_scenarios = []

    for category in ['banking', 'consulting', 'law']:
        filepath = Path(data_dir) / category / 'scenarios.json'

        try:
            with open(filepath, 'r') as f:
                scenarios = json.load(f)

            # Add category to each scenario
            for s in scenarios:
                s['category'] = category

            all_scenarios.extend(scenarios)
            print(f'Loaded {len(scenarios):2d} {category} scenarios from {filepath}')

        except FileNotFoundError:
            print(f'  {filepath} not found — skipping')

    print(f'Total: {len(all_scenarios)} scenarios')
    return all_scenarios


def format_prompt(scenario):
    """Format a scenario into a prompt for the model."""
    workspace = '=== WORKSPACE FILES ===\n\n'
    for filename, content in scenario['files'].items():
        workspace += f'--- {filename} ---\n{content}\n\n'

    return (
        f'{workspace}\n'
        f'=== YOUR TASK ===\n'
        f'{scenario["task"]}\n\n'
        f'Review all files carefully. Produce a professional, complete response.\n'
    )


print("=" * 60)
print("LOADING SCENARIOS")
print("=" * 60)
scenarios = load_all_scenarios()
print()


# ============================================================
# GPU CHECK
# ============================================================
print("=" * 60)
print("GPU INFORMATION")
print("=" * 60)
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print("WARNING: No CUDA GPU detected!")
print()


# ============================================================
# LOAD SMALLER MODEL (3B - Model being evaluated)
# ============================================================
print("=" * 60)
print("LOADING SMALLER MODEL (3B)")
print("=" * 60)

print(f'Loading model from: {SMALL_MODEL_PATH}...')

smaller_tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_PATH)
smaller_model = AutoModelForCausalLM.from_pretrained(
    SMALL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
smaller_model.eval()
print(f'✓ Smaller model loaded — {sum(p.numel() for p in smaller_model.parameters())/1e9:.1f}B params')
print()


# ============================================================
# LOAD BIGGER MODEL (14B - LLM-as-Judge)
# ============================================================
print("=" * 60)
print("LOADING BIGGER MODEL (14B) - JUDGE")
print("=" * 60)

BIGGER_MODEL_NAME = 'Qwen/Qwen2.5-14B-Instruct'
print(f'Loading {BIGGER_MODEL_NAME}...')

bigger_tokenizer = AutoTokenizer.from_pretrained(BIGGER_MODEL_NAME)
bigger_model = AutoModelForCausalLM.from_pretrained(
    BIGGER_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
bigger_model.eval()
print(f'✓ Bigger model loaded — {sum(p.numel() for p in bigger_model.parameters())/1e9:.1f}B params')
print(f'  This model will be used as LLM-as-Judge')
print()


# ============================================================
# INFERENCE FUNCTIONS
# ============================================================
def run_inference_for_smaller_model(prompt: str, max_new_tokens: int = 1024) -> str:
    """Run inference with the 3B model being evaluated."""
    messages = [
        {
            'role': 'system',
            'content': (
                'You are a highly skilled professional analyst. '
                'Review the provided workspace files carefully and '
                'respond with accurate, professional analysis.'
            )
        },
        {'role': 'user', 'content': prompt}
    ]
    text = smaller_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = smaller_tokenizer([text], return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = smaller_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=smaller_tokenizer.eos_token_id
        )
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return smaller_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_inference_for_bigger_model(prompt: str, max_new_tokens: int = 1024) -> str:
    """Run inference with the 14B judge model."""
    messages = [
        {
            'role': 'system',
            'content': (
                'You are a highly skilled professional analyst. '
                'Review the provided workspace files carefully and '
                'respond with accurate, professional analysis.'
            )
        },
        {'role': 'user', 'content': prompt}
    ]
    text = bigger_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = bigger_tokenizer([text], return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = bigger_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=bigger_tokenizer.eos_token_id
        )
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return bigger_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# SANITY CHECKS
# ============================================================
print("=" * 60)
print("SANITY CHECKS")
print("=" * 60)
print('Smaller model sanity check:', run_inference_for_smaller_model('What is 2+2? Answer in one word.'))
print('Bigger model sanity check:', run_inference_for_bigger_model('What is 2+2? Answer in one word.'))
print()


# ============================================================
# JUDGE FUNCTION
# ============================================================
def score_response(scenario: dict, response: str) -> dict:
    """
    Score a model response against the rubric using Qwen 14B as judge.
    Returns criteria scores (0/1 per criterion) and total score (0.0-1.0).
    """
    rubric_text = '\n'.join(
        f'{i+1}. {c}' for i, c in enumerate(scenario['rubric'])
    )
    n = len(scenario['rubric'])

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
        # Use bigger model as judge
        judge_response = run_inference_for_bigger_model(judge_prompt, max_new_tokens=256)

        # Strip markdown code fences if present
        raw = judge_response.strip()
        if '```' in raw:
            raw = raw.split('```')[1].lstrip('json').strip()

        result = json.loads(raw)
        scores = result['scores'][:n]  # Guard against extra items

        return {
            'criteria_scores': scores,
            'criteria_met':    sum(scores),
            'criteria_total':  n,
            'total_score':     sum(scores) / n,
            'reasoning':       result.get('reasoning', '')
        }

    except Exception as e:
        print(f'  Judge error: {e}')
        return {
            'criteria_scores': [0] * n,
            'criteria_met':    0,
            'criteria_total':  n,
            'total_score':     0.0,
            'reasoning':       f'Error: {e}'
        }


print(f'Judge ready: {BIGGER_MODEL_NAME} (local model)')
print()


# ============================================================
# BASELINE EVALUATION
# ============================================================
def run_baseline_eval(scenarios, max_scenarios=None):
    """Run evaluation on all scenarios."""
    results = []
    eval_set = scenarios[:max_scenarios] if max_scenarios else scenarios
    total = len(eval_set)

    print(f'Running baseline on {total} scenarios')
    print(f'Started: {datetime.now().strftime("%H:%M:%S")}')
    print('-' * 55)

    for i, scenario in enumerate(eval_set):
        print(f'[{i+1}/{total}] {scenario["id"]} ({scenario["category"]}, {scenario.get("difficulty", "N/A")})')

        prompt   = format_prompt(scenario)
        t0       = time.time()
        response = run_inference_for_smaller_model(prompt)
        elapsed  = time.time() - t0

        scored   = score_response(scenario, response)

        results.append({
            'id':         scenario['id'],
            'category':   scenario['category'],
            'difficulty': scenario.get('difficulty', 'N/A'),
            'world':      scenario.get('world', 'N/A'),
            'response':   response,
            'elapsed_s':  round(elapsed, 1),
            **scored
        })

        pct = scored['total_score'] * 100
        print(f'  Score: {scored["criteria_met"]}/{scored["criteria_total"]} ({pct:.0f}%)  [{elapsed:.1f}s]')
        print(f'  {scored["reasoning"][:90]}')
        print()
        time.sleep(0.3)  # Small delay between scenarios

    return results


print("=" * 60)
print("RUNNING BASELINE EVALUATION")
print("=" * 60)
# Run on all scenarios (remove max_scenarios parameter to run all 22)
baseline_results = run_baseline_eval(scenarios)
print()


# ============================================================
# ANALYZE RESULTS
# ============================================================
def analyze_results(results, label='Baseline'):
    """Analyze and print results summary."""
    n = len(results)
    avg   = sum(r['total_score'] for r in results) / n
    pass1 = sum(1 for r in results if r['total_score'] == 1.0) / n

    print(f'\n{"="*55}')
    print(f'  {label} — {MODEL_NAME}')
    print(f'{"="*55}')
    print(f'  Scenarios:     {n}')
    print(f'  Avg score:     {avg*100:.1f}%')
    print(f'  Pass@1 (100%): {pass1*100:.1f}%')
    print()

    for cat in ['banking', 'consulting', 'law']:
        sub = [r for r in results if r['category'] == cat]
        if sub:
            cat_avg = sum(r['total_score'] for r in sub) / len(sub)
            print(f'  {cat.upper():12s}  {cat_avg*100:.1f}%  ({len(sub)} scenarios)')

    print()
    worst = sorted(results, key=lambda r: r['total_score'])[:3]
    print('  Lowest scores (most room to improve):')
    for r in worst:
        print(f'    {r["id"]:15s} {r["total_score"]*100:.0f}%  {r["world"]}')

    return {'avg_score': avg, 'pass_at_1': pass1, 'n': n}


print("=" * 60)
print("ANALYSIS")
print("=" * 60)
summary = analyze_results(baseline_results, 'BASELINE')
print()


# ============================================================
# SAVE RESULTS
# ============================================================
# Save to project root
output_file = Path(__file__).parent.parent / 'baseline_results.json'

with open(output_file, 'w') as f:
    json.dump({
        'model':          MODEL_NAME,
        'model_path':     SMALL_MODEL_PATH,
        'judge':          'Qwen/Qwen2.5-14B-Instruct (local)',
        'timestamp':      datetime.now().isoformat(),
        'summary':        summary,
        'results':        baseline_results
    }, f, indent=2)

print(f'✓ Saved to {output_file}')
print('  Keep this file — compare it after training to show improvement')
print()


# ============================================================
# INSPECT WORST RESPONSE
# ============================================================
print("=" * 60)
print("WORST PERFORMING SCENARIO")
print("=" * 60)

worst = min(baseline_results, key=lambda r: r['total_score'])

print(f'Worst: {worst["id"]} — {worst["world"]}')
print(f'Score: {worst["criteria_met"]}/{worst["criteria_total"]} ({worst["total_score"]*100:.0f}%)')
print()
print('MODEL RESPONSE:')
print('-' * 50)
print(worst['response'])
print()
print('RUBRIC BREAKDOWN:')

# Re-load scenario to get rubric text
scenario = next(s for s in scenarios if s['id'] == worst['id'])
for criterion, score in zip(scenario['rubric'], worst['criteria_scores']):
    icon = '✓' if score == 1 else '✗'
    print(f'  [{icon}] {criterion}')

print()
print(f'Judge note: {worst["reasoning"]}')
print()


# ============================================================
# NEXT STEPS
# ============================================================
print("=" * 60)
print("NEXT STEPS")
print("=" * 60)
print('  1. Note your baseline avg score')
print('  2. Train with GRPO: python training/train_grpo.py')
print('  3. Update MODEL_NAME in .env to fine-tuned model (e.g., Qwen2.5-3B-Instruct_v0)')
print('  4. Re-run this script: python training/baseline_eval.py')
print('  5. Compare results: improvement = post-training - baseline')
print()
print("✅ Done!")
