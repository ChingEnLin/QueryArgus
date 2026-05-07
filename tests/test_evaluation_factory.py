"""Tests for the evaluator factory: profile + override → concrete evaluators."""

from __future__ import annotations

import pytest

from queryargus.agent.evaluation import (
    CompositeFindingEvaluator,
    CompositeRunEvaluator,
    JudgeRunEvaluator,
    RulesActionEvaluator,
    RulesFindingEvaluator,
    RulesRunEvaluator,
    SelfFindingEvaluator,
    SelfRunEvaluator,
    build_action_evaluator,
    build_finding_evaluator,
    build_run_evaluator,
)
from queryargus.llm.client import ScriptedLLMClient
from queryargus.models.config import (
    PROFILE_BALANCED,
    PROFILE_FAST,
    PROFILE_THOROUGH,
    EvaluatorConfig,
)


def _agent_llm() -> ScriptedLLMClient:
    return ScriptedLLMClient()


def test_profile_fast_is_rules_only() -> None:
    cfg = PROFILE_FAST
    llm = _agent_llm()
    assert isinstance(build_action_evaluator(cfg), RulesActionEvaluator)
    assert isinstance(build_finding_evaluator(cfg, agent_llm=llm), RulesFindingEvaluator)
    assert isinstance(build_run_evaluator(cfg, agent_llm=llm), RulesRunEvaluator)


def test_profile_balanced_uses_composite_finding_and_self_run() -> None:
    cfg = PROFILE_BALANCED
    llm = _agent_llm()
    finding = build_finding_evaluator(cfg, agent_llm=llm)
    run = build_run_evaluator(cfg, agent_llm=llm)
    assert isinstance(finding, CompositeFindingEvaluator)
    # Composite is rules-then-self.
    assert any(isinstance(e, RulesFindingEvaluator) for e in finding.evaluators)
    assert any(isinstance(e, SelfFindingEvaluator) for e in finding.evaluators)
    assert isinstance(run, SelfRunEvaluator)


def test_profile_thorough_with_judge_llm() -> None:
    cfg = PROFILE_THOROUGH
    llm = _agent_llm()
    judge_llm = _agent_llm()
    run = build_run_evaluator(cfg, agent_llm=llm, judge_llm=judge_llm, judge_model_name="judge-x")
    assert isinstance(run, JudgeRunEvaluator)


def test_judge_strategy_without_judge_llm_raises() -> None:
    cfg = EvaluatorConfig(run_evaluator="judge")
    with pytest.raises(ValueError, match="judge_llm"):
        build_run_evaluator(cfg, agent_llm=_agent_llm())


def test_none_strategy_returns_none() -> None:
    cfg = EvaluatorConfig(action_evaluator="none", finding_evaluator="none", run_evaluator="none")
    llm = _agent_llm()
    assert build_action_evaluator(cfg) is None
    assert build_finding_evaluator(cfg, agent_llm=llm) is None
    assert build_run_evaluator(cfg, agent_llm=llm) is None


def test_run_composite_with_judge_chains_three() -> None:
    cfg = EvaluatorConfig(run_evaluator="composite")
    llm = _agent_llm()
    run = build_run_evaluator(cfg, agent_llm=llm, judge_llm=_agent_llm())
    assert isinstance(run, CompositeRunEvaluator)
    assert len(run.evaluators) == 3  # rules + self + judge
