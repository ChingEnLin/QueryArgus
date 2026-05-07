"""CompositeEvaluator: chain semantics and verdict aggregation."""

from __future__ import annotations

from queryargus.agent.evaluation.base import ActionEvaluator
from queryargus.agent.evaluation.composite import CompositeActionEvaluator
from queryargus.agent.state import AgentState
from queryargus.models.action import AgentAction
from queryargus.models.evaluation import EvaluationResult, EvaluationVerdict


class _StubAction(ActionEvaluator):
    def __init__(self, verdict: EvaluationVerdict, name: str, calls: list[str]) -> None:
        self._v = verdict
        self._name = name
        self._calls = calls

    def evaluate(self, action: AgentAction, state: AgentState) -> EvaluationResult:
        self._calls.append(self._name)
        return EvaluationResult(
            verdict=self._v,
            score={EvaluationVerdict.PASS: 1.0, EvaluationVerdict.WARN: 0.5, EvaluationVerdict.FAIL: 0.0}[self._v],
            reason=f"{self._name} returned {self._v.value}",
            evaluated_by=self._name,
            critique=None if self._v == EvaluationVerdict.PASS else f"critique-from-{self._name}",
        )


def _state() -> AgentState:
    return AgentState(collection="c", database="d", cosmos_account="a", iteration_budget=20)


def _action() -> AgentAction:
    return AgentAction(reasoning="x", action="schema_sample", action_input={}, confidence=0.9)


def test_composite_short_circuits_on_first_fail() -> None:
    calls: list[str] = []
    composite = CompositeActionEvaluator(
        evaluators=[
            _StubAction(EvaluationVerdict.PASS, "first", calls),
            _StubAction(EvaluationVerdict.FAIL, "second", calls),
            _StubAction(EvaluationVerdict.PASS, "third", calls),
        ]
    )
    result = composite.evaluate(_action(), _state())
    assert calls == ["first", "second"]  # third never runs
    assert result.verdict == EvaluationVerdict.FAIL


def test_composite_accumulates_warn_when_no_fail() -> None:
    calls: list[str] = []
    composite = CompositeActionEvaluator(
        evaluators=[
            _StubAction(EvaluationVerdict.PASS, "first", calls),
            _StubAction(EvaluationVerdict.WARN, "second", calls),
            _StubAction(EvaluationVerdict.PASS, "third", calls),
        ]
    )
    result = composite.evaluate(_action(), _state())
    assert calls == ["first", "second", "third"]
    assert result.verdict == EvaluationVerdict.WARN


def test_composite_passes_when_all_pass() -> None:
    calls: list[str] = []
    composite = CompositeActionEvaluator(
        evaluators=[
            _StubAction(EvaluationVerdict.PASS, "first", calls),
            _StubAction(EvaluationVerdict.PASS, "second", calls),
        ]
    )
    result = composite.evaluate(_action(), _state())
    assert result.verdict == EvaluationVerdict.PASS


def test_composite_critique_concatenates() -> None:
    calls: list[str] = []
    composite = CompositeActionEvaluator(
        evaluators=[
            _StubAction(EvaluationVerdict.WARN, "first", calls),
            _StubAction(EvaluationVerdict.WARN, "second", calls),
        ]
    )
    result = composite.evaluate(_action(), _state())
    assert result.critique
    assert "critique-from-first" in result.critique
    assert "critique-from-second" in result.critique
