



import json
import os
from typing import Optional

from groq import Groq


class LLMJudge:
    """
    Scores agent responses using Llama 3.3 70B via Groq (free tier).
    Each rubric criterion is scored binary: 1 (met) or 0 (not met).
    Final reward = fraction of criteria met.
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
            )
        self.client = Groq(api_key=key)
        self.model  = "llama-3.3-70b-versatile"

    def score(self, scenario: dict, response: str) -> dict:
        n = len(scenario["rubric"])
        rubric_text = "\n".join(
            f"{i+1}. {c}" for i, c in enumerate(scenario["rubric"])
        )
        prompt = f"""You are an expert evaluator assessing a professional analysis.

TASK:
{scenario['task']}

AGENT RESPONSE:
{response}

RUBRIC — score each criterion 1 (met) or 0 (not met):
{rubric_text}

Rules: Be strict. Binary only, no partial credit.
Base scores only on what is explicitly stated in the response.

Respond ONLY with valid JSON, no other text:
{{"scores": [list of {n} integers each 0 or 1], "reasoning": "one concise sentence"}}"""

        try:
            resp = self.client.chat.completions.create(
                model    = self.model,
                messages = [{"role": "user", "content": prompt}],
                max_tokens  = 256,
                temperature = 0.0
            )
            raw = resp.choices[0].message.content.strip()
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
            print(e)
            return {
                "criteria_scores": [0] * n,
                "criteria_met":    0,
                "criteria_total":  n,
                "reward":          0.0,
                "reasoning":       f"Judge error: {e}"
            }