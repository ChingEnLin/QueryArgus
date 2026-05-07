"""Build the action/finding/run evaluator triple from an ``EvaluatorConfig``.

This is the single place that knows how to compose Rules + Self + Judge into
the strategies the spec defines. The CLI calls these to honour ``--eval-profile``
without anything else needing to grow conditional logic.

v1 limitations:
- Self-eval is supported on the *finding* and *run* gates, not on the action
  gate (per scope decision in self_eval.py).
- Judge-eval is run-level only.
"""

from __future__ import annotations

from queryargus.agent.evaluation.base import (
    ActionEvaluator,
    FindingEvaluator,
    RunEvaluator,
)
from queryargus.agent.evaluation.composite import (
    CompositeFindingEvaluator,
    CompositeRunEvaluator,
)
from queryargus.agent.evaluation.judge import JudgeRunEvaluator
from queryargus.agent.evaluation.rules import (
    RulesActionEvaluator,
    RulesFindingEvaluator,
    RulesRunEvaluator,
)
from queryargus.agent.evaluation.self_eval import SelfFindingEvaluator, SelfRunEvaluator
from queryargus.llm.client import LLMClient
from queryargus.models.config import EvaluatorConfig


def build_action_evaluator(
    config: EvaluatorConfig,
) -> ActionEvaluator | None:
    strategy = config.action_evaluator
    if strategy == "none":
        return None
    if strategy in ("rules", "composite"):
        # composite at the action gate degrades to rules-only in v1
        return RulesActionEvaluator()
    raise ValueError(
        f"Action gate strategy {strategy!r} is not supported in v1; use 'rules' or 'none'."
    )


def build_finding_evaluator(
    config: EvaluatorConfig,
    agent_llm: LLMClient,
    *,
    agent_model_name: str = "agent",
) -> FindingEvaluator | None:
    strategy = config.finding_evaluator
    if strategy == "none":
        return None
    if strategy == "rules":
        return RulesFindingEvaluator()
    if strategy == "self":
        return SelfFindingEvaluator(llm=agent_llm, model_name=agent_model_name)
    if strategy == "composite":
        return CompositeFindingEvaluator(
            evaluators=[
                RulesFindingEvaluator(),
                SelfFindingEvaluator(llm=agent_llm, model_name=agent_model_name),
            ]
        )
    raise ValueError(
        f"Finding gate strategy {strategy!r} is not supported in v1 (judge-finding not implemented)."
    )


def build_run_evaluator(
    config: EvaluatorConfig,
    agent_llm: LLMClient,
    *,
    agent_model_name: str = "agent",
    judge_llm: LLMClient | None = None,
    judge_model_name: str | None = None,
) -> RunEvaluator | None:
    strategy = config.run_evaluator
    if strategy == "none":
        return None
    if strategy == "rules":
        return RulesRunEvaluator()
    if strategy == "self":
        return SelfRunEvaluator(llm=agent_llm, model_name=agent_model_name)
    if strategy == "judge":
        if judge_llm is None:
            raise ValueError(
                "Run gate strategy 'judge' requires a separate judge LLM client; "
                "pass judge_llm=GeminiClient(model='gemini-2.5-pro') or similar."
            )
        return JudgeRunEvaluator(llm=judge_llm, model_name=judge_model_name or "judge")
    if strategy == "composite":
        evaluators: list[RunEvaluator] = [
            RulesRunEvaluator(),
            SelfRunEvaluator(llm=agent_llm, model_name=agent_model_name),
        ]
        if judge_llm is not None:
            evaluators.append(JudgeRunEvaluator(llm=judge_llm, model_name=judge_model_name or "judge"))
        return CompositeRunEvaluator(evaluators=evaluators)
    raise ValueError(f"Unknown run gate strategy: {strategy!r}")
