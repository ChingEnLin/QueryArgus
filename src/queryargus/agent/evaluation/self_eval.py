"""Self-evaluators — the agent LLM critiques its own output.

These re-use whichever ``LLMClient`` the agent runs on. Each call adds tokens
onto ``AgentState.total_usage`` so the audit report still surfaces the true
cost of the evaluation stack.

Implemented for finding and run gates only. Action-gate self-eval is a poor
trade-off (doubling LLM calls for marginal verdict quality) and is omitted.
Use ``RulesActionEvaluator`` for the action gate instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from queryargus.agent.evaluation.eval_prompts import (
    SELF_EVAL_SYSTEM,
    SELF_FINDING_USER_TEMPLATE,
    SELF_RUN_USER_TEMPLATE,
)
from queryargus.agent.evaluation.llm_parse import parse_evaluation_json
from queryargus.agent.state import AgentState
from queryargus.llm.client import LLMClient
from queryargus.models.evaluation import EvaluationResult
from queryargus.models.finding import Finding
from queryargus.models.report import AuditReport


@dataclass
class SelfFindingEvaluator:
    """Same-LLM critique of a proposed finding."""

    llm: LLMClient
    model_name: str = "self"

    def evaluate(self, finding: Finding, state: AgentState) -> EvaluationResult:
        user = SELF_FINDING_USER_TEMPLATE.format(
            finding_json=finding.model_dump_json(indent=2),
            state_summary=state.summarize(),
            model_name=self.model_name,
        )
        response = self.llm.complete_json(system=SELF_EVAL_SYSTEM, user=user)
        state.total_usage = state.total_usage + response.usage
        return parse_evaluation_json(response.raw, evaluated_by=f"self:{self.model_name}")


@dataclass
class SelfRunEvaluator:
    """Same-LLM critique of the final report."""

    llm: LLMClient
    model_name: str = "self"

    def evaluate(self, report: AuditReport, state: AgentState) -> EvaluationResult:
        user = SELF_RUN_USER_TEMPLATE.format(
            report_json=report.model_dump_json(indent=2, exclude={"run_trace"}),
            run_trace_json=_truncate_trace(report),
            model_name=self.model_name,
        )
        response = self.llm.complete_json(system=SELF_EVAL_SYSTEM, user=user)
        state.total_usage = state.total_usage + response.usage
        return parse_evaluation_json(response.raw, evaluated_by=f"self:{self.model_name}")


def _truncate_trace(report: AuditReport, *, max_actions: int = 30) -> str:
    """Show the last N actions to keep the prompt bounded on long runs."""
    actions = report.run_trace[-max_actions:]
    return "[" + ",\n".join(a.model_dump_json() for a in actions) + "]"
