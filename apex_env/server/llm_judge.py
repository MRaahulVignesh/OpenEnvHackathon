



import json
import os
from typing import Optional
from pathlib import Path

from vllm import LLM, SamplingParams


class LLMJudge:
    """
    Scores agent responses using Qwen2.5-14B-Instruct via vLLM (local).
    Each rubric criterion is scored binary: 1 (met) or 0 (not met).
    Final reward = fraction of criteria met.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        gpu_memory_utilization: float = 0.3,
        tensor_parallel_size: int = 1
    ):
        """
        Initialize vLLM-based judge.

        Args:
            model_name: Model name or path. Defaults to Qwen/Qwen2.5-14B-Instruct
            gpu_memory_utilization: GPU memory fraction for vLLM (0.0-1.0)
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        self.model_name = model_name or os.getenv(
            "JUDGE_MODEL", "Qwen/Qwen2.5-14B-Instruct"
        )

        print(f"Loading judge model: {self.model_name}")
        print(f"  GPU memory utilization: {gpu_memory_utilization}")

        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="bfloat16",
        )

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            stop=None,
        )

        print(f"✓ Judge model loaded successfully")

    def score(self, scenario: dict, response: str) -> dict:
        """
        Score a response against the scenario rubric using vLLM.

        Args:
            scenario: Dict with 'task' and 'rubric' keys
            response: Agent's response to score

        Returns:
            Dict with criteria_scores, criteria_met, criteria_total, reward, reasoning
        """
        n = len(scenario["rubric"])
        rubric_text = "\n".join(
            f"{i+1}. {c}" for i, c in enumerate(scenario["rubric"])
        )

        # Build the prompt in chat format
        messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator assessing professional analysis responses."
            },
            {
                "role": "user",
                "content": f"""TASK:
{scenario['task']}

AGENT RESPONSE:
{response}

RUBRIC — score each criterion 1 (met) or 0 (not met):
{rubric_text}

Rules: Be strict. Binary only, no partial credit.
Base scores only on what is explicitly stated in the response.

Respond ONLY with valid JSON, no other text:
{{"scores": [list of {n} integers each 0 or 1], "reasoning": "one concise sentence"}}"""
            }
        ]

        # Format using chat template
        tokenizer = self.llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        try:
            # Generate with vLLM
            outputs = self.llm.generate([prompt], self.sampling_params)
            raw = outputs[0].outputs[0].text.strip()

            # Parse response
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()

            result = json.loads(raw)
            scores = result["scores"][:n]

            return {
                "criteria_scores": scores,
                "criteria_met":    sum(scores),
                "criteria_total":  n,
                "reward":          round(sum(scores) / n, 4),
                "reasoning":       result.get("reasoning", "")
            }

        except Exception as e:
            print(f"Judge error: {e}")
            return {
                "criteria_scores": [0] * n,
                "criteria_met":    0,
                "criteria_total":  n,
                "reward":          0.0,
                "reasoning":       f"Judge error: {e}"
            }