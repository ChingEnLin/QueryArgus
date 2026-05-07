"""LLMClient — the seam between the planner and a concrete LLM provider.

v1 ships ``GeminiClient`` only (per locked decision D3). Tests use the
in-memory ``ScriptedLLMClient`` to drive the loop deterministically without
hitting any network.

Every client returns an ``LLMResponse`` so the planner can accumulate token
usage across iterations — useful for cost visibility and for noticing prompt
bloat as the iteration history grows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

from queryargus.models.action import AgentAction


@dataclass(frozen=True)
class TokenUsage:
    """Per-call token counts. Zeros for clients that don't report usage."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass
class LLMResponse:
    action: AgentAction
    usage: TokenUsage = field(default_factory=TokenUsage)


class LLMClient(Protocol):
    """One method: turn a prompt into a structured action plus a usage record."""

    def propose_action(self, *, system: str, user: str) -> LLMResponse:
        ...


class ScriptedLLMClient:
    """A deterministic LLM stand-in for tests.

    Yields actions from a queue in FIFO order. If the queue is exhausted, the
    fallback action is returned (defaults to ``conclude``) so test loops always
    terminate. Records every prompt it receives so tests can assert on what the
    planner actually showed the model. Optional ``usage_per_call`` lets tests
    simulate token consumption (default zeros).
    """

    def __init__(
        self,
        actions: list[AgentAction],
        *,
        fallback: AgentAction | None = None,
        usage_per_call: TokenUsage | None = None,
    ) -> None:
        self._queue: deque[AgentAction] = deque(actions)
        self._fallback = fallback or AgentAction(
            reasoning="ScriptedLLMClient queue exhausted — concluding.",
            action="conclude",
            confidence=1.0,
        )
        self._usage = usage_per_call or TokenUsage()
        self.prompts: list[tuple[str, str]] = []

    def propose_action(self, *, system: str, user: str) -> LLMResponse:
        self.prompts.append((system, user))
        action = self._queue.popleft() if self._queue else self._fallback
        return LLMResponse(action=action, usage=self._usage)
