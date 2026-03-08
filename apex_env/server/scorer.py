
Copy

"""
RLScorer — thin wrapper around LLMJudge with gold score caching.

score()      — score an agent response against the rubric
score_gold() — score the scenario's gold_output (cached per scenario id)
"""

from apex_env.server.llm_judge import LLMJudge


class RLScorer:
    def __init__(self, judge: LLMJudge):
        self.judge = judge

    def score(self, scenario: dict, response: str) -> dict:
        return self.judge.score(scenario, response)

    def score_gold(self, scenario: dict) -> dict:
        """
        Score the scenario's gold_output.
        Delegates to LLMJudge.score_gold() which caches result by scenario id.
        """
        return self.judge.score_gold(scenario)