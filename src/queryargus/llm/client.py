"""LLMClient — the seam between the planner and a concrete LLM provider.

v1 ships ``GeminiClient`` only (per locked decision D3). Tests use the
in-memory ``ScriptedLLMClient`` to drive the loop deterministically without
hitting any network.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

from queryargus.models.action import AgentAction


class LLMClient(Protocol):
    """One method: turn a prompt into a structured action."""

    def propose_action(self, *, system: str, user: str) -> AgentAction:
        ...


class ScriptedLLMClient:
    """A deterministic LLM stand-in for tests.

    Yields actions from a queue in FIFO order. If the queue is exhausted, the
    fallback action is returned (defaults to ``conclude``) so test loops always
    terminate. Records every prompt it receives so tests can assert on what the
    planner actually showed the model.
    """

    def __init__(
        self,
        actions: list[AgentAction],
        *,
        fallback: AgentAction | None = None,
    ) -> None:
        self._queue: deque[AgentAction] = deque(actions)
        self._fallback = fallback or AgentAction(
            reasoning="ScriptedLLMClient queue exhausted — concluding.",
            action="conclude",
            confidence=1.0,
        )
        self.prompts: list[tuple[str, str]] = []

    def propose_action(self, *, system: str, user: str) -> AgentAction:
        self.prompts.append((system, user))
        if self._queue:
            return self._queue.popleft()
        return self._fallback
