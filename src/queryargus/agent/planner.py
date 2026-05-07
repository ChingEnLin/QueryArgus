"""Planner — build a prompt from current state, call the LLM, return AgentAction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from queryargus.agent.prompts import SYSTEM_PROMPT, render_user_prompt
from queryargus.agent.state import AgentState
from queryargus.llm.client import LLMClient
from queryargus.models.action import AgentAction

logger = logging.getLogger(__name__)


@dataclass
class Planner:
    """Stateless wrapper around an LLMClient that knows how to format the prompt."""

    llm: LLMClient

    def propose(self, state: AgentState) -> AgentAction:
        user = render_user_prompt(state.summarize())
        response = self.llm.propose_action(system=SYSTEM_PROMPT, user=user)
        state.total_usage = state.total_usage + response.usage
        state.usage_per_iteration.append(response.usage)
        logger.info(
            "planner iteration=%d proposed=%s confidence=%.2f tokens=%d (in=%d out=%d) total=%d",
            state.iteration,
            response.action.action,
            response.action.confidence,
            response.usage.total_tokens,
            response.usage.input_tokens,
            response.usage.output_tokens,
            state.total_usage.total_tokens,
        )
        return response.action
