"""Tests for self/judge LLM-backed evaluators."""

from __future__ import annotations

from queryargus.agent.evaluation.judge import JudgeRunEvaluator
from queryargus.agent.evaluation.llm_parse import parse_evaluation_json
from queryargus.agent.evaluation.self_eval import (
    SelfFindingEvaluator,
    SelfRunEvaluator,
)
from queryargus.agent.state import AgentState
from queryargus.llm.client import ScriptedLLMClient, TokenUsage
from queryargus.models.evaluation import EvaluationVerdict
from queryargus.models.finding import Finding, FindingSeverity
from queryargus.models.report import AuditReport


def _state() -> AgentState:
    return AgentState(collection="c", database="d", cosmos_account="a", iteration_budget=20)


def _finding(**kw: object) -> Finding:
    base = {
        "field": "user.age",
        "category": "outlier_value",
        "severity": FindingSeverity.HIGH,
        "description": "Some users have impossible ages.",
        "hypothesis": "Sentinel value 9999",
        "evidence_query": "{age: {$gt: 150}}",
        "affected_count": 5,
        "affected_pct": 0.05,
    }
    base.update(kw)
    return Finding(**base)  # type: ignore[arg-type]


def _report() -> AuditReport:
    return AuditReport(
        collection="c", database="d", cosmos_account="a",
        duration_seconds=1.0, documents_sampled=100, collection_size=1000,
    )


def test_parse_evaluation_json_clean() -> None:
    raw = '{"verdict": "pass", "score": 0.9, "reason": "looks fine", "evaluated_by": "self:test"}'
    result = parse_evaluation_json(raw, evaluated_by="self:test")
    assert result.verdict == EvaluationVerdict.PASS
    assert result.score == 0.9


def test_parse_evaluation_json_strips_markdown_fence() -> None:
    raw = '```json\n{"verdict": "fail", "score": 0.1, "reason": "no evidence", "evaluated_by": "x"}\n```'
    result = parse_evaluation_json(raw, evaluated_by="self:test")
    assert result.verdict == EvaluationVerdict.FAIL


def test_parse_evaluation_json_fills_evaluated_by_when_missing() -> None:
    raw = '{"verdict": "warn", "score": 0.5, "reason": "borderline"}'
    result = parse_evaluation_json(raw, evaluated_by="self:test")
    assert result.evaluated_by == "self:test"


def test_parse_evaluation_json_malformed_falls_back_to_warn() -> None:
    result = parse_evaluation_json("not json at all", evaluated_by="self:test")
    assert result.verdict == EvaluationVerdict.WARN
    assert "unparseable" in result.reason


def test_self_finding_evaluator_accumulates_tokens_onto_state() -> None:
    state = _state()
    llm = ScriptedLLMClient(
        json_responses=['{"verdict": "pass", "score": 0.92, "reason": "evidence is strong", "evaluated_by": "self:test"}'],
        usage_per_call=TokenUsage(input_tokens=200, output_tokens=40),
    )
    evaluator = SelfFindingEvaluator(llm=llm, model_name="test")
    result = evaluator.evaluate(_finding(), state)

    assert result.verdict == EvaluationVerdict.PASS
    assert result.evaluated_by == "self:test"
    assert state.total_usage.input_tokens == 200
    assert state.total_usage.output_tokens == 40


def test_self_run_evaluator_uses_run_template() -> None:
    state = _state()
    llm = ScriptedLLMClient(
        json_responses=['{"verdict": "warn", "score": 0.55, "reason": "shallow coverage", "evaluated_by": "self:gemini"}']
    )
    evaluator = SelfRunEvaluator(llm=llm, model_name="gemini")
    result = evaluator.evaluate(_report(), state)
    assert result.verdict == EvaluationVerdict.WARN
    # The user prompt should mention the report — confirms we used the run template.
    assert llm.json_prompts
    _system, user = llm.json_prompts[-1]
    assert "AUDIT REPORT" in user


def test_judge_run_evaluator_accumulates_separate_tokens() -> None:
    state = _state()
    llm = ScriptedLLMClient(
        json_responses=['{"verdict": "fail", "score": 0.2, "reason": "weak", "critique": "more coverage", "evaluated_by": "judge:gpt"}'],
        usage_per_call=TokenUsage(input_tokens=500, output_tokens=80),
    )
    evaluator = JudgeRunEvaluator(llm=llm, model_name="gpt")
    result = evaluator.evaluate(_report(), state)
    assert result.verdict == EvaluationVerdict.FAIL
    assert result.evaluated_by == "judge:gpt"
    # Judge tokens land on state too — single shared cost line in the report.
    assert state.total_usage.input_tokens == 500
    assert state.total_usage.output_tokens == 80
