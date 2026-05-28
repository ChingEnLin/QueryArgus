"""RunObserver Protocol + null / multi implementations.

Observers are duck-typed: they only need to implement the hooks they care
about. ``MultiObserver`` checks ``hasattr`` before calling each hook on each
child so partial observers work cleanly and one bad observer cannot break a
run — child exceptions are caught and logged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from queryargus.llm.client import TokenUsage
    from queryargus.models.action import AgentAction
    from queryargus.models.finding import Finding
    from queryargus.models.report import AuditReport

logger = logging.getLogger(__name__)


@runtime_checkable
class RunObserver(Protocol):
    def on_run_start(self, *, run_id: UUID, collection: str) -> None: ...
    def on_iteration_start(self, *, iter: int) -> None: ...
    def on_llm_call(
        self,
        *,
        purpose: Literal["propose_action", "self_eval", "judge"],
        model: str,
        usage: TokenUsage,
        latency_ms: int,
    ) -> None: ...
    def on_tool_call(
        self,
        *,
        name: str,
        args_summary: str,
        ok: bool,
        latency_ms: int,
        error: str | None,
    ) -> None: ...
    def on_action(self, *, action: AgentAction) -> None: ...
    def on_finding(self, *, finding: Finding) -> None: ...
    def on_eval(
        self,
        *,
        target: Literal["action", "finding", "run"],
        verdict: str,
        score: float,
        evaluator: str,
    ) -> None: ...
    def on_run_complete(self, *, report: AuditReport) -> None: ...


class NullObserver:
    """No-op observer. The default when callers don't pass one."""

    def on_run_start(self, *, run_id: UUID, collection: str) -> None: ...
    def on_iteration_start(self, *, iter: int) -> None: ...
    def on_llm_call(
        self,
        *,
        purpose: Literal["propose_action", "self_eval", "judge"],
        model: str,
        usage: TokenUsage,
        latency_ms: int,
    ) -> None: ...
    def on_tool_call(
        self,
        *,
        name: str,
        args_summary: str,
        ok: bool,
        latency_ms: int,
        error: str | None,
    ) -> None: ...
    def on_action(self, *, action: AgentAction) -> None: ...
    def on_finding(self, *, finding: Finding) -> None: ...
    def on_eval(
        self,
        *,
        target: Literal["action", "finding", "run"],
        verdict: str,
        score: float,
        evaluator: str,
    ) -> None: ...
    def on_run_complete(self, *, report: AuditReport) -> None: ...


class MultiObserver:
    """Fan out hook calls to a list of observers, isolating per-observer failures."""

    def __init__(self, observers: list[Any]) -> None:
        self._observers: list[Any] = list(observers)

    def _dispatch(self, hook: str, /, **kwargs: Any) -> None:
        for obs in self._observers:
            fn = getattr(obs, hook, None)
            if fn is None:
                continue
            try:
                fn(**kwargs)
            except Exception:
                logger.exception(
                    "observer %s raised in %s", type(obs).__name__, hook
                )

    def on_run_start(self, *, run_id: UUID, collection: str) -> None:
        self._dispatch("on_run_start", run_id=run_id, collection=collection)

    def on_iteration_start(self, *, iter: int) -> None:
        self._dispatch("on_iteration_start", iter=iter)

    def on_llm_call(
        self,
        *,
        purpose: Literal["propose_action", "self_eval", "judge"],
        model: str,
        usage: TokenUsage,
        latency_ms: int,
    ) -> None:
        self._dispatch(
            "on_llm_call",
            purpose=purpose,
            model=model,
            usage=usage,
            latency_ms=latency_ms,
        )

    def on_tool_call(
        self,
        *,
        name: str,
        args_summary: str,
        ok: bool,
        latency_ms: int,
        error: str | None,
    ) -> None:
        self._dispatch(
            "on_tool_call",
            name=name,
            args_summary=args_summary,
            ok=ok,
            latency_ms=latency_ms,
            error=error,
        )

    def on_action(self, *, action: AgentAction) -> None:
        self._dispatch("on_action", action=action)

    def on_finding(self, *, finding: Finding) -> None:
        self._dispatch("on_finding", finding=finding)

    def on_eval(
        self,
        *,
        target: Literal["action", "finding", "run"],
        verdict: str,
        score: float,
        evaluator: str,
    ) -> None:
        self._dispatch(
            "on_eval",
            target=target,
            verdict=verdict,
            score=score,
            evaluator=evaluator,
        )

    def on_run_complete(self, *, report: AuditReport) -> None:
        self._dispatch("on_run_complete", report=report)
