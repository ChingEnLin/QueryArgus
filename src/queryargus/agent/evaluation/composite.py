"""Composite evaluators — chain multiple evaluators of the same kind.

Runs evaluators in order. Stops at the first ``FAIL``. Otherwise accumulates
verdicts: a single ``WARN`` anywhere in the chain demotes the verdict to
``WARN``. The final ``EvaluationResult`` carries every reason concatenated and
every critique joined with ``" | "``.

Used to layer cheap deterministic rules in front of expensive LLM evaluators
(weekend 3): rules first, self-eval next, judge last.
"""

from __future__ import annotations

from dataclasses import dataclass

from queryargus.agent.evaluation.base import (
    ActionEvaluator,
    FindingEvaluator,
    RunEvaluator,
)
from queryargus.agent.state import AgentState
from queryargus.models.action import AgentAction
from queryargus.models.evaluation import EvaluationResult, EvaluationVerdict
from queryargus.models.finding import Finding
from queryargus.models.report import AuditReport


def _merge(results: list[EvaluationResult], evaluated_by: str) -> EvaluationResult:
    if not results:
        return EvaluationResult(
            verdict=EvaluationVerdict.PASS,
            score=1.0,
            reason="composite: no evaluators",
            evaluated_by=evaluated_by,
        )
    # Worst verdict wins.
    rank = {EvaluationVerdict.PASS: 0, EvaluationVerdict.WARN: 1, EvaluationVerdict.FAIL: 2}
    verdict = max((r.verdict for r in results), key=lambda v: rank[v])
    score = min(r.score for r in results)
    reasons = " | ".join(f"[{r.evaluated_by}] {r.reason}" for r in results)
    critiques = [r.critique for r in results if r.critique]
    return EvaluationResult(
        verdict=verdict,
        score=score,
        reason=reasons,
        critique=" | ".join(critiques) if critiques else None,
        evaluated_by=evaluated_by,
    )


@dataclass
class CompositeActionEvaluator:
    evaluators: list[ActionEvaluator]
    evaluated_by: str = "composite"

    def evaluate(self, action: AgentAction, state: AgentState) -> EvaluationResult:
        results: list[EvaluationResult] = []
        for ev in self.evaluators:
            r = ev.evaluate(action, state)
            results.append(r)
            if r.verdict == EvaluationVerdict.FAIL:
                break
        return _merge(results, self.evaluated_by)


@dataclass
class CompositeFindingEvaluator:
    evaluators: list[FindingEvaluator]
    evaluated_by: str = "composite"

    def evaluate(self, finding: Finding, state: AgentState) -> EvaluationResult:
        results: list[EvaluationResult] = []
        for ev in self.evaluators:
            r = ev.evaluate(finding, state)
            results.append(r)
            if r.verdict == EvaluationVerdict.FAIL:
                break
        return _merge(results, self.evaluated_by)


@dataclass
class CompositeRunEvaluator:
    evaluators: list[RunEvaluator]
    evaluated_by: str = "composite"

    def evaluate(self, report: AuditReport, state: AgentState) -> EvaluationResult:
        results: list[EvaluationResult] = []
        for ev in self.evaluators:
            r = ev.evaluate(report, state)
            results.append(r)
            if r.verdict == EvaluationVerdict.FAIL:
                break
        return _merge(results, self.evaluated_by)
