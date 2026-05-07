"""Judge evaluator — a separate, more capable LLM grades the final report.

The judge is intentionally run-level only: per-action or per-finding judging
costs too much for the marginal signal. See spec §5.4C.

The judge LLM is independent of the agent LLM (different model, possibly
different provider). Tokens are accumulated onto ``AgentState.total_usage``
just like self-eval — so cost visibility is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

from queryargus.agent.evaluation.eval_prompts import (
    JUDGE_RUN_USER_TEMPLATE,
    JUDGE_SYSTEM,
)
from queryargus.agent.evaluation.llm_parse import parse_evaluation_json
from queryargus.agent.state import AgentState
from queryargus.llm.client import LLMClient
from queryargus.models.evaluation import EvaluationResult
from queryargus.models.report import AuditReport


@dataclass
class JudgeRunEvaluator:
    """Independent-LLM grade for the final audit report."""

    llm: LLMClient
    model_name: str = "judge"

    def evaluate(self, report: AuditReport, state: AgentState) -> EvaluationResult:
        user = JUDGE_RUN_USER_TEMPLATE.format(
            report_json=report.model_dump_json(indent=2, exclude={"run_trace"}),
            run_trace_json=_full_trace(report),
            model_name=self.model_name,
        )
        response = self.llm.complete_json(system=JUDGE_SYSTEM, user=user)
        state.total_usage = state.total_usage + response.usage
        return parse_evaluation_json(response.raw, evaluated_by=f"judge:{self.model_name}")


def _full_trace(report: AuditReport) -> str:
    return "[" + ",\n".join(a.model_dump_json() for a in report.run_trace) + "]"
