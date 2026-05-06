"""AgentAction — a single LLM-proposed step in the planning loop."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ActionName = Literal["schema_sample", "run_query", "get_stats", "write_finding", "conclude"]


class AgentAction(BaseModel):
    """A structured action the LLM proposes per ReAct iteration."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(min_length=1, description="Why the agent is taking this action.")
    action: ActionName
    action_input: dict[str, Any] = Field(default_factory=dict, description="Tool-specific parameters.")
    confidence: float = Field(ge=0.0, le=1.0)
