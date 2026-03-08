"""
APEX Professional Tasks Environment

Extends openenv.core Environment to train agents on investment banking,
management consulting, and corporate law tasks — targeting the Mercor
APEX-Agents benchmark.

The environment:
  - Loads 22 professional scenarios from data/
  - Serves one scenario per episode via reset()
  - Scores agent responses against rubrics via Groq LLM judge
  - Returns reward = fraction of rubric criteria met (0.0 – 1.0)
"""

import json
import random
from pathlib import Path
from typing import Optional
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State, EnvironmentMetadata

from apex_env.server.llm_judge import LLMJudge
from apex_env.models import APEXAction, APEXObservation


# ── Scenario helpers ───────────────────────────────────────────────────────────

def _load_scenarios(data_dir: str = "data") -> list[dict]:
    """Load all scenarios from data/banking, data/consulting, data/law."""
    all_scenarios = []
    base = Path(data_dir)

    for category in ["banking", "consulting", "law"]:
        path = base / category / "scenarios.json"
        if path.exists():
            with open(path) as f:
                scenarios = json.load(f)
            for s in scenarios:
                s["category"] = category
            all_scenarios.extend(scenarios)

    if not all_scenarios:
        raise FileNotFoundError(
            f"No scenarios found in '{data_dir}'. "
            "Expected subfolders: banking/, consulting/, law/"
        )
    return all_scenarios


def _format_prompt(scenario: dict) -> str:
    """Build the text prompt the agent sees: workspace files + task."""
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

# ── Environment ────────────────────────────────────────────────────────────────

class APEXEnvironment(Environment):
    """
    APEX Professional Tasks Environment — extends openenv.core.Environment.

    Each episode = one scenario (single-step).
    The agent gets the workspace files + task via reset(),
    submits a response via step(), and receives a reward (0.0–1.0).

    Usage:
        env = APEXEnvironment()
        obs = env.reset()          # get a scenario
        obs = env.step(action)     # submit response, obs.reward is set
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        data_dir:        str           = "apex_env/data",
        category_filter: Optional[str] = None,
        groq_api_key:    Optional[str] = None,
        shuffle:         bool          = True,
    ):
        super().__init__()

        # Load scenarios
        all_scenarios = _load_scenarios(data_dir)
        if category_filter:
            all_scenarios = [s for s in all_scenarios if s["category"] == category_filter]
            if not all_scenarios:
                raise ValueError(f"No scenarios for category: {category_filter}")

        self.scenarios       = all_scenarios
        self.category_filter = category_filter
        self.shuffle         = shuffle
        self.judge           = LLMJudge(api_key=groq_api_key)

        # Episode state
        self._state          = State(episode_id=str(uuid4()), step_count=0)
        self._current        = None
        self._queue          = []
        self._scenarios_seen = 0

    @property
    def state(self) -> State:
        """Return current episode state."""
        return self._state

    def _reset_rubric(self):
        """Reset rubric-related state for new episode."""
        pass

    def _apply_transform(self, obs: APEXObservation) -> APEXObservation:
        """Apply any transformations to the observation before returning it."""
        return obs

    def reset(
        self,
        seed:       Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> APEXObservation:
        """
        Start a new episode. Picks the next scenario and returns it.
        """
        print("In Reset")
        if seed is not None:
            random.seed(seed)

        # Refill queue when empty
        if not self._queue:
            self._queue = self.scenarios.copy()
            if self.shuffle:
                random.shuffle(self._queue)

        self._current = self._queue.pop()
        self._state   = State(
            episode_id = episode_id or str(uuid4()),
            step_count = 0
        )
        self._reset_rubric()

        return APEXObservation(
            scenario_id = self._current["id"],
            category    = self._current["category"],
            world       = self._current["world"],
            prompt      = _format_prompt(self._current),
            step        = 0,
            done        = False,
            reward      = None,
        )

    def step(
        self,
        action:    APEXAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> APEXObservation:
        """
        Score the agent's response against the rubric.
        Returns observation with reward set.
        """
        print("In Step")
        if self._current is None:
            raise RuntimeError("Call reset() before step()")

        self._state.step_count  += 1
        self._scenarios_seen    += 1

        scored = self.judge.score(self._current, action.response)

        obs = APEXObservation(
            scenario_id     = self._current["id"],
            category        = self._current["category"],
            world           = self._current["world"],
            prompt          = _format_prompt(self._current),
            step            = self._state.step_count,
            done            = True,          # single-step episodes
            reward          = scored["reward"],
            criteria_scores = scored["criteria_scores"],
            criteria_met    = scored["criteria_met"],
            criteria_total  = scored["criteria_total"],
            reasoning       = scored["reasoning"],
            metadata        = {
                "difficulty": self._current["difficulty"],
                "rubric":     self._current["rubric"],
            }
        )
        return self._apply_transform(obs)

    def get_metadata(self):
        return EnvironmentMetadata(
            name        = "APEX Professional Tasks",
            description = (
                "RL training environment for investment banking, management consulting, "
                "and corporate law tasks. Targets the Mercor APEX-Agents benchmark."
            ),
            version = "1.0.0",
            author  = "OpenENV Hackathon",
        )