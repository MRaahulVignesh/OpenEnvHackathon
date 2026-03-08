"""
Data models for the APEX Professional Tasks Environment.
Extends OpenENV's base Action and Observation types.
"""

from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class APEXAction(Action):
    """The agent submits a professional response to the current task."""
    response: str = Field(..., description="The agent's professional response to the task")


class APEXObservation(Observation):
    """
    Observation from the APEX environment.
    Contains the current scenario + scoring breakdown after step().
    """
    # ── Scenario info ──────────────────────────────────────────────────────────
    scenario_id: str  = Field(default="", description="Unique scenario identifier")
    category:    str  = Field(default="", description="banking | consulting | law")
    world:       str  = Field(default="", description="Scenario world/company name")
    prompt:      str  = Field(default="", description="Workspace files + task instruction")
    step:        int  = Field(default=0,  description="Current step in episode")
    difficulty:  str  = Field(default="medium", description="easy | medium | hard")

    # ── Rubric scores (populated after step(), empty after reset()) ────────────
    criteria_scores: list[int] = Field(default=[], description="Per-criterion scores (0 or 1)")
    criteria_met:    int       = Field(default=0,  description="Number of criteria met")
    criteria_total:  int       = Field(default=0,  description="Total number of criteria")
    reasoning:       str       = Field(default="", description="Judge's reasoning")

    # ── Reward breakdown ───────────────────────────────────────────────────────
    base_reward:    float = Field(default=0.0, description="Rubric score (criteria_met / criteria_total)")
    noise_bonus:    float = Field(default=0.0, description="+0.2 if agent detected injected noise")
    gold_score:     float = Field(default=0.0, description="Judge score of the gold reference response")
    peer_reward:    float = Field(default=0.5, description="Relative reward vs gold: 0.5=matches, >0.5=beats gold")
    blended_reward: float = Field(default=0.0, description="60% peer + 40% absolute rubric before noise bonus")

    # ── Adversarial noise ──────────────────────────────────────────────────────
    noise_injected: bool = Field(default=False, description="Whether noise was injected this episode")
    noise_detected: bool = Field(default=False, description="Whether agent flagged the noise")

    # ── Difficulty progression ─────────────────────────────────────────────────
    tier_status: dict = Field(
        default_factory=dict,
        description="Current difficulty tier per category e.g. {'banking': 'medium', 'law': 'easy'}"
    )