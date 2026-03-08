"""
RLScorer - Scoring interface for APEX environment
"""

import os
from pathlib import Path
from apex_env.server.llm_judge import LLMJudge


class RLScorer:
    """
    Scorer for agent responses in APEX environment.
    Uses LLMJudge internally, can be extended with additional scoring methods.
    """

    def __init__(self):
        """Initialize scorer with LLM judge (reads config from env vars)."""
        judge_model = os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        models_folder = os.getenv("MODELS_FOLDER", "./models")
        gpu_memory = float(os.getenv("JUDGE_GPU_MEMORY", "0.2"))
        tensor_parallel = int(os.getenv("JUDGE_TENSOR_PARALLEL", "1"))

        # Resolve local path
        if "/" in judge_model:
            model_short_name = judge_model.split("/")[-1]
            local_path = Path(models_folder) / "base_model" / model_short_name
            if not (local_path / "config.json").exists():
                raise FileNotFoundError(
                    f"Judge model not found at {local_path}. "
                    f"Run: python training/download_model.py"
                )
            model_name = str(local_path)
        else:
            model_name = judge_model

        self.llm_judge = LLMJudge(
            model_name=model_name,
            gpu_memory_utilization=gpu_memory,
            tensor_parallel_size=tensor_parallel
        )

    def score(self, scenario: dict, response: str) -> dict:
        """
        Score a response against the scenario rubric.

        Args:
            scenario: Dict with 'task' and 'rubric' keys
            response: Agent's response to score

        Returns:
            Dict with criteria_scores, criteria_met, criteria_total, reward, reasoning
        """
        return self.llm_judge.score(scenario, response)
