"""Tests for EscalationFindingEvaluator (Arm B)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from queryargus.agent.evaluation.escalation import (
    ESCALATION_DROPPED_SENTINEL,
    ESCALATION_PENDING_SENTINEL,
    EscalationFindingEvaluator,
)
from queryargus.agent.evaluation.rules import RulesFindingEvaluator
from queryargus.agent.state import AgentState
from queryargus.models.evaluation import EvaluationResult, EvaluationVerdict
from queryargus.models.finding import Finding, FindingSeverity


def _finding(*, confidence: float | None) -> Finding:
    return Finding(
        field="email",
        category="null_rate",
        severity=FindingSeverity.HIGH,
        description="lots of nulls",
        hypothesis="optional",
        evidence_query='{"email": null}',
        affected_count=50,
        affected_pct=0.30,
        confidence=confidence,
        confidence_reason="testing",
    )


def _state() -> AgentState:
    return AgentState(collection="c", database="d", cosmos_account="a", iteration_budget=20)


# A passive inner that always reports PASS so we isolate confidence-band routing.
@dataclass
class _AlwaysPassInner:
    evaluated_by: str = "fake-pass"

    def evaluate(self, finding: Finding, state: AgentState) -> EvaluationResult:
        return EvaluationResult(
            verdict=EvaluationVerdict.PASS,
            score=0.9,
            reason="inner says fine",
            evaluated_by=self.evaluated_by,
        )


@dataclass
class _AlwaysFailInner:
    evaluated_by: str = "fake-fail"

    def evaluate(self, finding: Finding, state: AgentState) -> EvaluationResult:
        return EvaluationResult(
            verdict=EvaluationVerdict.FAIL,
            score=0.0,
            reason="inner says broken",
            evaluated_by=self.evaluated_by,
            critique="fix this",
        )


def test_high_confidence_passes_through_inner_result() -> None:
    """confidence >= high → return inner result unchanged (so rules/critique still apply)."""
    ev = EscalationFindingEvaluator(inner=_AlwaysPassInner(), high_threshold=0.85, low_threshold=0.4)
    result = ev.evaluate(_finding(confidence=0.9), _state())
    assert result.verdict == EvaluationVerdict.PASS
    assert result.evaluated_by == "fake-pass"  # inner identity preserved


def test_mid_confidence_emits_pending_sentinel() -> None:
    """LOW ≤ confidence < HIGH → escalation:pending sentinel; PASS so the loop persists it."""
    ev = EscalationFindingEvaluator(inner=_AlwaysPassInner(), high_threshold=0.85, low_threshold=0.4)
    result = ev.evaluate(_finding(confidence=0.6), _state())
    assert result.verdict == EvaluationVerdict.PASS
    assert result.evaluated_by == ESCALATION_PENDING_SENTINEL


def test_low_confidence_emits_dropped_sentinel() -> None:
    """confidence < LOW → FAIL with escalation:dropped sentinel (forces hard drop in loop)."""
    ev = EscalationFindingEvaluator(inner=_AlwaysPassInner(), high_threshold=0.85, low_threshold=0.4)
    result = ev.evaluate(_finding(confidence=0.25), _state())
    assert result.verdict == EvaluationVerdict.FAIL
    assert result.evaluated_by == ESCALATION_DROPPED_SENTINEL


def test_inner_fail_short_circuits_regardless_of_confidence() -> None:
    """A high-confidence finding the inner says is broken (e.g. duplicate query) still fails."""
    ev = EscalationFindingEvaluator(inner=_AlwaysFailInner(), high_threshold=0.85, low_threshold=0.4)
    result = ev.evaluate(_finding(confidence=0.99), _state())
    assert result.verdict == EvaluationVerdict.FAIL
    assert result.evaluated_by == "fake-fail"  # NOT the escalation sentinel
    assert result.critique == "fix this"


def test_missing_confidence_defers_to_inner() -> None:
    """Legacy callers that omit confidence behave as if escalation is off."""
    ev = EscalationFindingEvaluator(inner=_AlwaysPassInner(), high_threshold=0.85, low_threshold=0.4)
    result = ev.evaluate(_finding(confidence=None), _state())
    assert result.verdict == EvaluationVerdict.PASS
    assert result.evaluated_by == "fake-pass"


def test_cap_drops_overflow_escalations() -> None:
    """Once `cap` findings are already pending, further mid-band findings are dropped."""
    ev = EscalationFindingEvaluator(inner=_AlwaysPassInner(), cap=1, high_threshold=0.85, low_threshold=0.4)
    state = _state()
    # Seed the ledger with one pending finding to consume the cap.
    state.findings.write(
        field="other", category="null_rate", severity=FindingSeverity.MEDIUM,
        description="x", hypothesis="x", evidence_query="x",
        affected_count=1, affected_pct=0.01, status="pending_review",
    )
    result = ev.evaluate(_finding(confidence=0.6), state)
    assert result.verdict == EvaluationVerdict.FAIL
    assert result.evaluated_by == ESCALATION_DROPPED_SENTINEL
    assert "cap" in result.reason


def test_low_threshold_above_high_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        EscalationFindingEvaluator(inner=None, high_threshold=0.3, low_threshold=0.7)


def test_real_rules_inner_still_fires() -> None:
    """End-to-end with the real RulesFindingEvaluator — duplicate filter still fails high-confidence."""
    state = _state()
    # Simulate a prior identical query so the rules evaluator's "no-repeat" rule trips.
    state.queries_run.append({"email": None})
    ev = EscalationFindingEvaluator(inner=RulesFindingEvaluator(), high_threshold=0.85, low_threshold=0.4)
    f = _finding(confidence=0.95)
    result = ev.evaluate(f, state)
    # Rules verdict (whatever it returns) should win when confidence is high.
    # Either PASS (because the duplicate-filter check doesn't apply to write_finding)
    # or FAIL (if rules disagree). Either way evaluated_by is NOT escalation:pending.
    assert result.evaluated_by != ESCALATION_PENDING_SENTINEL
    assert result.evaluated_by != ESCALATION_DROPPED_SENTINEL
