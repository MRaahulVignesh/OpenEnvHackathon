"""
LLMJudge — scores agent responses using a local model via transformers.

Gold score caching: score_gold() scores a scenario's gold_output once and
caches the result by scenario id. Subsequent calls return the cached value,
so peer comparison adds zero extra judge calls during training.

Uses transformers pipeline (NOT vLLM) so it coexists with the training
vLLM instance on the same GPU without CUDA allocator conflicts.

The judge loads once at server startup, then runs inference synchronously.
Each call scores one response against one scenario's rubric.
"""

import json
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMJudge:
    """
    Scores agent responses using a local model via transformers (not vLLM).

    Why transformers instead of vLLM?
      Training uses vLLM in "colocate" mode, which takes over the CUDA
      allocator during generation. A second vLLM instance (the judge) will
      silently OOM and return reward=0. transformers' eager mode uses a
      separate CUDA stream and coexists safely.

    Memory usage: ~14GB for Qwen2.5-7B-Instruct in bfloat16.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        gpu_memory_utilization: float = 0.2,   # ignored, kept for API compat
        tensor_parallel_size: int = 1,          # ignored, kept for API compat
    ):
        self.model_name = model_name or os.getenv(
            "JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct"
        )

        print(f"Loading judge model (transformers): {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        self.model.eval()
        self._gold_cache: dict[str, dict] = {}  # scenario_id → scored result

        print("✓ Judge model loaded (transformers, bfloat16, cuda:0)")

    def score_gold(self, scenario: dict) -> dict:
        """
        Score the scenario's gold_output against its rubric.
        Result is cached by scenario id — only judged once per scenario
        across the entire training run (zero extra cost after first call).
        """
        sid = scenario["id"]
        if sid not in self._gold_cache:
            gold = scenario.get("gold_output", "")
            if not gold:
                n = len(scenario["rubric"])
                self._gold_cache[sid] = {
                    "criteria_scores": [1] * n,
                    "criteria_met":    n,
                    "criteria_total":  n,
                    "reward":          1.0,
                    "reasoning":       "No gold output — assuming perfect score",
                }
            else:
                print(f"  📐 Scoring gold output for [{sid}] (cached after this)")
                self._gold_cache[sid] = self.score(scenario, gold)
        return self._gold_cache[sid]

    @torch.inference_mode()
    def score(self, scenario: dict, response: str) -> dict:
        """
        Score a response against the scenario rubric.

        Args:
            scenario: Dict with 'task' and 'rubric' keys
            response: Agent's response to score

        Returns:
            Dict with criteria_scores, criteria_met, criteria_total,
            reward (0.0–1.0), reasoning
        """
        n = len(scenario["rubric"])
        rubric_text = "\n".join(
            f"{i+1}. {c}" for i, c in enumerate(scenario["rubric"])
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator assessing professional analysis responses.",
            },
            {
                "role": "user",
                "content": (
                    f"TASK:\n{scenario['task']}\n\n"
                    f"AGENT RESPONSE:\n{response}\n\n"
                    f"RUBRIC — score each criterion 1 (met) or 0 (not met):\n{rubric_text}\n\n"
                    f"Rules: Be strict. Binary only, no partial credit.\n"
                    f"Base scores only on what is explicitly stated in the response.\n"
                    f"Respond ONLY with valid JSON, no other text:\n"
                    f'{{"scores": [list of {n} integers each 0 or 1], "reasoning": "one concise sentence"}}'
                ),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,            # greedy — deterministic judge
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Strip markdown fences if present
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()

            result = json.loads(raw)
            scores = [int(s) for s in result["scores"][:n]]

            return {
                "criteria_scores": scores,
                "criteria_met":    sum(scores),
                "criteria_total":  n,
                "reward":          round(sum(scores) / n, 4),
                "reasoning":       result.get("reasoning", ""),
            }

        except Exception as e:
            print(f"Judge error: {e}\nRaw output: {raw if 'raw' in dir() else 'N/A'}")
            return {
                "criteria_scores": [0] * n,
                "criteria_met":    0,
                "criteria_total":  n,
                "reward":          0.0,
                "reasoning":       f"Judge error: {e}",
            }