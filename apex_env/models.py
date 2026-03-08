"""
Data models for the APEX Professional Tasks Environment.

Extends OpenENV's base Action and Observation types.
"""

from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class APEXAction(Action):
    """
    Action for the APEX environment.
    The agent submits a professional response to the current task.
    """
    response: str = Field(..., description="The agent's professional response to the task")


class APEXObservation(Observation):
    """
    Observation from the APEX environment.
    Contains the current scenario the agent must respond to.
    """
    scenario_id:  str            = Field(default="",    description="Unique scenario identifier")
    category:     str            = Field(default="",    description="banking | consulting | law")
    world:        str            = Field(default="",    description="Scenario world/company name")
    prompt:       str            = Field(default="",    description="Workspace files + task instruction")
    step:         int            = Field(default=0,     description="Current step in episode")
    # Rubric breakdown (populated after step(), empty after reset())
    criteria_scores: list[int]   = Field(default=[],   description="Per-criterion scores (0 or 1)")
    criteria_met:    int         = Field(default=0,     description="Number of criteria met")
    criteria_total:  int         = Field(default=0,     description="Total number of criteria")
    reasoning:       str         = Field(default="",    description="Judge's reasoning")