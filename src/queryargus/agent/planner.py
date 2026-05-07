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
        action = self.llm.propose_action(system=SYSTEM_PROMPT, user=user)
        logger.info(
            "planner iteration=%d proposed=%s confidence=%.2f",
            state.iteration,
            action.action,
            action.confidence,
        )
        return action
