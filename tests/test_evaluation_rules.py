"""Per-rule tests for the rules-based evaluators."""

from __future__ import annotations

from queryargus.agent.evaluation.rules import (
    RulesActionEvaluator,
    RulesFindingEvaluator,
    RulesRunEvaluator,
)
from queryargus.agent.state import AgentState
from queryargus.models.action import AgentAction
from queryargus.models.evaluation import EvaluationVerdict
from queryargus.models.finding import Finding, FindingSeverity
from queryargus.models.report import AuditReport
from queryargus.tools.schema_sample import FieldStats, SchemaSampleResult


def _state(**kw: object) -> AgentState:
    base = {
        "collection": "users",
        "database": "db",
        "cosmos_account": "acct",
        "iteration_budget": 20,
    }
    base.update(kw)
    return AgentState(**base)  # type: ignore[arg-type]


def _make_schema(*paths: str) -> SchemaSampleResult:
    return SchemaSampleResult(
        collection="users",
        documents_sampled=100,
        fields=[FieldStats(path=p, present_count=100, missing_count=0, types={"str": 100}) for p in paths],
        truncated_paths=[],
    )


# ----------------------------------------------------------------------------
# Action rules
# ----------------------------------------------------------------------------

def test_action_no_repeat_query_fails_on_duplicate() -> None:
    state = _state()
    state.schema = _make_schema("age")
    state.queries_run.append({"age": {"$gt": 150}})

    action = AgentAction(
        reasoning="check outliers", action="run_query",
        action_input={"filter": {"age": {"$gt": 150}}}, confidence=0.8,
    )
    result = RulesActionEvaluator().evaluate(action, state)
    assert result.verdict == EvaluationVerdict.FAIL
    assert "no_repeat_query" in result.reason


def test_action_no_repeat_query_passes_on_new_filter() -> None:
    state = _state()
    state.schema = _make_schema("age")
    state.queries_run.append({"age": {"$gt": 150}})

    action = AgentAction(
        reasoning="check different bucket", action="run_query",
        action_input={"filter": {"age": {"$lt": 0}}}, confidence=0.8,
    )
    assert RulesActionEvaluator().evaluate(action, state).verdict == EvaluationVerdict.PASS


def test_action_sample_before_query_fails_when_no_schema() -> None:
    state = _state()  # schema=None
    action = AgentAction(
        reasoning="early query", action="run_query",
        action_input={"filter": {"x": 1}}, confidence=0.5,
    )
    result = RulesActionEvaluator().evaluate(action, state)
    assert result.verdict == EvaluationVerdict.FAIL
    assert "sample_before_query" in result.reason


def test_action_sample_before_query_allows_schema_sample_first() -> None:
    state = _state()
    action = AgentAction(reasoning="survey", action="schema_sample", action_input={"sample_size": 200}, confidence=0.9)
    assert RulesActionEvaluator().evaluate(action, state).verdict == EvaluationVerdict.PASS


def test_action_input_shape_run_query_requires_dict_filter() -> None:
    state = _state()
    state.schema = _make_schema("age")
    bad = AgentAction(
        reasoning="malformed", action="run_query",
        action_input={"filter": "not-a-dict"}, confidence=0.5,
    )
    assert RulesActionEvaluator().evaluate(bad, state).verdict == EvaluationVerdict.FAIL


def test_action_input_shape_get_stats_invalid_op() -> None:
    state = _state()
    state.schema = _make_schema("age")
    bad = AgentAction(
        reasoning="bad op", action="get_stats",
        action_input={"field": "age", "operation": "median"}, confidence=0.5,
    )
    assert RulesActionEvaluator().evaluate(bad, state).verdict == EvaluationVerdict.FAIL


def test_action_input_shape_write_finding_missing_required() -> None:
    state = _state()
    state.schema = _make_schema("age")
    bad = AgentAction(
        reasoning="incomplete", action="write_finding",
        action_input={"field": "age"}, confidence=0.7,
    )
    assert RulesActionEvaluator().evaluate(bad, state).verdict == EvaluationVerdict.FAIL


# ----------------------------------------------------------------------------
# Finding rules
# ----------------------------------------------------------------------------

def _finding(**kw: object) -> Finding:
    base = {
        "field": "user.age",
        "category": "outlier_value",
        "severity": FindingSeverity.HIGH,
        "description": "Some users have impossible ages.",
        "hypothesis": "Sentinel value 9999 is being written",
        "evidence_query": "{age: {$gt: 150}}",
        "affected_count": 5,
        "affected_pct": 0.05,
        "sample_values": [9999],
    }
    base.update(kw)
    return Finding(**base)  # type: ignore[arg-type]


def test_finding_evidence_required_fails_at_zero() -> None:
    s = _state()
    s.schema = _make_schema("age")
    f = _finding(affected_count=0, affected_pct=0.0)
    result = RulesFindingEvaluator().evaluate(f, s)
    assert result.verdict == EvaluationVerdict.FAIL
    assert "evidence_required" in result.reason


def test_finding_severity_critical_under_threshold_fails() -> None:
    s = _state()
    f = _finding(severity=FindingSeverity.CRITICAL, affected_pct=0.005)  # 0.5% < 1%
    result = RulesFindingEvaluator().evaluate(f, s)
    assert result.verdict == EvaluationVerdict.FAIL
    assert "severity_calibration_critical" in result.reason


def test_finding_severity_high_marginal_warns() -> None:
    s = _state()
    f = _finding(severity=FindingSeverity.HIGH, affected_pct=0.0005)  # 0.05% < 0.1%
    result = RulesFindingEvaluator().evaluate(f, s)
    assert result.verdict == EvaluationVerdict.WARN


def test_finding_evidence_query_required() -> None:
    s = _state()
    f = _finding(evidence_query="   ")
    result = RulesFindingEvaluator().evaluate(f, s)
    assert result.verdict == EvaluationVerdict.FAIL


def test_finding_passes_clean_input() -> None:
    s = _state()
    assert RulesFindingEvaluator().evaluate(_finding(), s).verdict == EvaluationVerdict.PASS


# ----------------------------------------------------------------------------
# Run rules
# ----------------------------------------------------------------------------

def _report(**kw: object) -> AuditReport:
    base = {
        "collection": "users",
        "database": "db",
        "cosmos_account": "acct",
        "duration_seconds": 1.0,
        "documents_sampled": 100,
        "collection_size": 1000,
    }
    base.update(kw)
    return AuditReport(**base)  # type: ignore[arg-type]


def test_run_minimum_field_coverage_fails_when_under_50pct() -> None:
    s = _state()
    s.iteration = 5
    s.schema = _make_schema("a", "b", "c", "d")
    s.fields_investigated.add("a")  # 1/4 = 25%
    result = RulesRunEvaluator().evaluate(_report(), s)
    assert result.verdict == EvaluationVerdict.FAIL


def test_run_minimum_field_coverage_passes_at_50pct() -> None:
    s = _state()
    s.iteration = 5
    s.schema = _make_schema("a", "b", "c", "d")
    s.fields_investigated |= {"a", "b"}  # 2/4 = 50%, not <50% so pass
    result = RulesRunEvaluator().evaluate(_report(), s)
    assert result.verdict != EvaluationVerdict.FAIL


def test_run_early_termination_fails_under_3() -> None:
    s = _state()
    s.iteration = 2
    s.schema = _make_schema("a", "b")
    s.fields_investigated |= {"a", "b"}
    result = RulesRunEvaluator().evaluate(_report(), s)
    assert result.verdict == EvaluationVerdict.FAIL
    assert "early_termination" in result.reason


def test_run_no_findings_on_large_sample_warns() -> None:
    s = _state()
    s.iteration = 10
    s.documents_sampled = 5000
    s.schema = _make_schema("a", "b")
    s.fields_investigated |= {"a", "b"}
    result = RulesRunEvaluator().evaluate(_report(documents_sampled=5000, collection_size=5000), s)
    # Findings empty; should at least WARN.
    assert result.verdict in {EvaluationVerdict.WARN, EvaluationVerdict.FAIL}
