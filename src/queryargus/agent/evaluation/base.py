"""Evaluator protocols and verdict combinators."""

from __future__ import annotations

from typing import Protocol

from queryargus.agent.state import AgentState
from queryargus.models.action import AgentAction
from queryargus.models.evaluation import EvaluationResult, EvaluationVerdict
from queryargus.models.finding import Finding
from queryargus.models.report import AuditReport

_RANK = {EvaluationVerdict.PASS: 0, EvaluationVerdict.WARN: 1, EvaluationVerdict.FAIL: 2}


def worst(verdicts: list[EvaluationVerdict]) -> EvaluationVerdict:
    """Pick the most severe verdict from a list (FAIL > WARN > PASS)."""
    if not verdicts:
        return EvaluationVerdict.PASS
    return max(verdicts, key=lambda v: _RANK[v])


class ActionEvaluator(Protocol):
    """Decide whether a proposed action is worth executing."""

    def evaluate(self, action: AgentAction, state: AgentState) -> EvaluationResult:
        ...


class FindingEvaluator(Protocol):
    """Decide whether a finding has enough evidence to commit."""

    def evaluate(self, finding: Finding, state: AgentState) -> EvaluationResult:
        ...


class RunEvaluator(Protocol):
    """Decide whether the audit run produced a satisfactory report."""

    def evaluate(self, report: AuditReport, state: AgentState) -> EvaluationResult:
        ...
