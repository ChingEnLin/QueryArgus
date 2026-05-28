from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from queryargus.llm.client import TokenUsage
from queryargus.models.action import AgentAction
from queryargus.models.finding import Finding, FindingSeverity
from queryargus.observability.observer import MultiObserver, NullObserver, RunObserver


@dataclass
class RecordingObserver:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def on_run_start(self, *, run_id: Any, collection: str) -> None:
        self.calls.append(("on_run_start", {"run_id": run_id, "collection": collection}))

    def on_iteration_start(self, *, iter: int) -> None:
        self.calls.append(("on_iteration_start", {"iter": iter}))

    def on_llm_call(self, *, purpose: str, model: str, usage: TokenUsage, latency_ms: int) -> None:
        self.calls.append(
            (
                "on_llm_call",
                {"purpose": purpose, "model": model, "usage": usage, "latency_ms": latency_ms},
            )
        )

    def on_tool_call(
        self, *, name: str, args_summary: str, ok: bool, latency_ms: int, error: str | None
    ) -> None:
        self.calls.append(
            (
                "on_tool_call",
                {
                    "name": name,
                    "args_summary": args_summary,
                    "ok": ok,
                    "latency_ms": latency_ms,
                    "error": error,
                },
            )
        )

    def on_action(self, *, action: AgentAction) -> None:
        self.calls.append(("on_action", {"action": action}))

    def on_finding(self, *, finding: Finding) -> None:
        self.calls.append(("on_finding", {"finding": finding}))

    def on_eval(self, *, target: str, verdict: str, score: float, evaluator: str) -> None:
        self.calls.append(
            (
                "on_eval",
                {"target": target, "verdict": verdict, "score": score, "evaluator": evaluator},
            )
        )

    def on_run_complete(self, *, report: Any) -> None:
        self.calls.append(("on_run_complete", {"report": report}))


def _sample_finding() -> Finding:
    return Finding(
        field="user.email",
        category="null_rate",
        severity=FindingSeverity.HIGH,
        description="elevated null rate",
        hypothesis="ingestion regression",
        evidence_query="db.users.count({email: null})",
        affected_count=10,
        affected_pct=0.1,
    )


def test_null_observer_accepts_every_hook() -> None:
    obs: RunObserver = NullObserver()
    obs.on_run_start(run_id=uuid4(), collection="c")
    obs.on_iteration_start(iter=1)
    obs.on_llm_call(
        purpose="propose_action", model="m", usage=TokenUsage(1, 2), latency_ms=10
    )
    obs.on_tool_call(name="t", args_summary="x", ok=True, latency_ms=5, error=None)
    obs.on_action(action=AgentAction(reasoning="r", action="conclude", confidence=1.0))
    obs.on_eval(target="action", verdict="pass", score=1.0, evaluator="rules")
    obs.on_finding(finding=_sample_finding())


def test_multi_observer_fans_out_in_order() -> None:
    a, b = RecordingObserver(), RecordingObserver()
    multi = MultiObserver([a, b])
    multi.on_iteration_start(iter=3)
    assert a.calls == [("on_iteration_start", {"iter": 3})]
    assert b.calls == [("on_iteration_start", {"iter": 3})]


def test_multi_observer_swallows_observer_exceptions(caplog: Any) -> None:
    class Boom:
        def on_iteration_start(self, *, iter: int) -> None:
            raise RuntimeError("boom")

    good = RecordingObserver()
    multi = MultiObserver([Boom(), good])
    with caplog.at_level(logging.ERROR, logger="queryargus.observability.observer"):
        multi.on_iteration_start(iter=7)
    assert good.calls == [("on_iteration_start", {"iter": 7})]
    assert any("Boom" in rec.message or "boom" in rec.message for rec in caplog.records)


def test_multi_observer_handles_partial_observers() -> None:
    class Partial:
        seen: Finding | None = None

        def on_finding(self, *, finding: Finding) -> None:
            self.seen = finding

    p = Partial()
    multi = MultiObserver([p])
    multi.on_iteration_start(iter=1)  # not defined on Partial — must not raise
    finding = _sample_finding()
    multi.on_finding(finding=finding)
    assert p.seen is finding
