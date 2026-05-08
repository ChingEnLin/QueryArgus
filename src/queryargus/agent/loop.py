"""ArgusAgent — the ReAct planning loop.

One iteration of the loop:

1. Planner proposes an AgentAction from current state.
2. Action evaluator gates the action (FAIL → critique into state, no tool call).
3. Tool dispatch executes the action against the live CosmosConnection.
4. State is updated with the observation.
5. If the action was ``write_finding``, the finding evaluator gates the commit
   (FAIL → dismissed, with the rejected_finding_policy applied).
6. If the action was ``conclude``, run evaluator gates the report (FAIL →
   the loop continues if budget remains, per run_fail_policy).

The loop terminates when ``conclude`` is accepted, the budget is exhausted, or
an unrecoverable error occurs. The final ``AuditReport`` always includes the
full run trace and every evaluation decision.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from queryargus.agent.evaluation.base import (
    ActionEvaluator,
    FindingEvaluator,
    RunEvaluator,
)
from queryargus.agent.evaluation.factory import (
    build_action_evaluator,
    build_finding_evaluator,
    build_run_evaluator,
)
from queryargus.agent.evaluation.rules import (
    RulesActionEvaluator,
    RulesFindingEvaluator,
    RulesRunEvaluator,
)
from queryargus.agent.planner import Planner
from queryargus.agent.state import AgentState
from queryargus.llm.client import LLMClient
from queryargus.models.action import AgentAction
from queryargus.models.config import ArgusConfig
from queryargus.models.connection import CosmosConnection
from queryargus.models.evaluation import (
    EvaluationRecord,
    EvaluationResult,
    EvaluationVerdict,
)
from queryargus.models.finding import Finding, FindingSeverity
from queryargus.models.history import HistoricalContext
from queryargus.models.report import AuditReport
from queryargus.tools.get_stats import get_stats
from queryargus.tools.run_query import run_query
from queryargus.tools.schema_sample import schema_sample

logger = logging.getLogger(__name__)


@dataclass
class ArgusAgent:
    """Top-level orchestration. Build via ``ArgusAgent.with_defaults`` or directly."""

    config: ArgusConfig
    llm: LLMClient
    action_evaluator: ActionEvaluator | None = None
    finding_evaluator: FindingEvaluator | None = None
    run_evaluator: RunEvaluator | None = None

    @classmethod
    def with_defaults(cls, config: ArgusConfig, llm: LLMClient) -> ArgusAgent:
        """Build an agent with the rules-only evaluation stack (PROFILE_FAST)."""
        return cls(
            config=config,
            llm=llm,
            action_evaluator=RulesActionEvaluator(),
            finding_evaluator=RulesFindingEvaluator(),
            run_evaluator=RulesRunEvaluator(),
        )

    @classmethod
    def from_config(
        cls,
        config: ArgusConfig,
        llm: LLMClient,
        *,
        agent_model_name: str | None = None,
        judge_llm: LLMClient | None = None,
        judge_model_name: str | None = None,
    ) -> ArgusAgent:
        """Build an agent with the evaluator stack derived from ``config.evaluation``."""
        agent_name = agent_model_name or config.llm_model
        return cls(
            config=config,
            llm=llm,
            action_evaluator=build_action_evaluator(config.evaluation),
            finding_evaluator=build_finding_evaluator(
                config.evaluation,
                agent_llm=llm,
                agent_model_name=agent_name,
            ),
            run_evaluator=build_run_evaluator(
                config.evaluation,
                agent_llm=llm,
                agent_model_name=agent_name,
                judge_llm=judge_llm,
                judge_model_name=judge_model_name,
            ),
        )

    def run(
        self,
        connection: CosmosConnection,
        collection: str,
        *,
        history: HistoricalContext | None = None,
    ) -> AuditReport:
        started = time.time()
        state = AgentState(
            collection=collection,
            database=connection.database_name,
            cosmos_account=connection.cosmos_account,
            iteration_budget=self.config.max_iterations,
            collection_size=_safe_collection_size(connection, collection),
            historical_context=history,
        )
        if history is not None and not history.is_empty:
            logger.info(
                "loaded historical context: runs=%d persistent_findings=%d one_off=%d dismissed=%d",
                history.runs_considered,
                len(history.persistent_findings),
                len(history.one_off_findings),
                len(history.dismissed_pairs),
            )
        planner = Planner(llm=self.llm)
        run_evaluation: EvaluationResult | None = None
        concluded = False

        while state.iteration < state.iteration_budget:
            state.iteration += 1
            action = planner.propose(state)
            state.history.append(action)
            state.last_critique = None  # consumed by the planner this iteration

            action_verdict = self._evaluate_action(action, state)
            if action_verdict and action_verdict.verdict == EvaluationVerdict.FAIL:
                logger.warning(
                    "action gate FAIL iter=%d action=%s reason=%s",
                    state.iteration, action.action, action_verdict.reason,
                )
                state.last_critique = action_verdict.critique or action_verdict.reason
                continue

            try:
                _dispatch_action(action, state, connection, collection, self)
            except _ConcludeRequested:
                concluded = True
                run_evaluation = self._evaluate_run(state, started)
                if (
                    run_evaluation
                    and run_evaluation.verdict == EvaluationVerdict.FAIL
                    and self.config.evaluation.run_fail_policy == "continue"
                    and state.remaining_budget > 0
                ):
                    state.last_critique = run_evaluation.critique or run_evaluation.reason
                    state.history[-1] = action.model_copy(
                        update={"reasoning": action.reasoning + " [run gate rejected; continuing]"}
                    )
                    concluded = False
                    continue
                break
            except Exception as exc:  # noqa: BLE001 — surface any tool-side error in the trace
                logger.exception("tool execution failed iter=%d action=%s", state.iteration, action.action)
                state.last_observation = f"ERROR executing {action.action}: {exc}"
                state.last_critique = (
                    f"the previous action raised an exception ({type(exc).__name__}); "
                    "try a different approach or conclude"
                )

        if not concluded:
            run_evaluation = self._evaluate_run(state, started)

        report = _build_report(state, started, run_evaluation)
        logger.info(
            "run done collection=%s findings=%d dismissed=%d iterations=%d duration=%.2fs",
            collection,
            len(report.findings),
            len(report.dismissed_findings),
            state.iteration,
            report.duration_seconds,
        )
        return report

    def _evaluate_action(self, action: AgentAction, state: AgentState) -> EvaluationResult | None:
        if self.action_evaluator is None:
            return None
        result = self.action_evaluator.evaluate(action, state)
        state.evaluation_records.append(
            EvaluationRecord(
                gate="action",
                evaluated_by=result.evaluated_by,
                verdict=result.verdict,
                score=result.score,
                reason=result.reason,
                critique=result.critique,
                iteration=state.iteration,
            )
        )
        return result

    def _evaluate_finding(self, finding: Finding, state: AgentState) -> EvaluationResult | None:
        if self.finding_evaluator is None:
            return None
        result = self.finding_evaluator.evaluate(finding, state)
        state.evaluation_records.append(
            EvaluationRecord(
                gate="finding",
                evaluated_by=result.evaluated_by,
                verdict=result.verdict,
                score=result.score,
                reason=result.reason,
                critique=result.critique,
                target_id=finding.id,
                iteration=state.iteration,
            )
        )
        return result

    def _evaluate_run(self, state: AgentState, started: float) -> EvaluationResult | None:
        if self.run_evaluator is None:
            return None
        # Build a partial report for the run evaluator's view.
        partial = _build_report(state, started, run_evaluation=None)
        result = self.run_evaluator.evaluate(partial, state)
        state.evaluation_records.append(
            EvaluationRecord(
                gate="run",
                evaluated_by=result.evaluated_by,
                verdict=result.verdict,
                score=result.score,
                reason=result.reason,
                critique=result.critique,
                iteration=state.iteration,
            )
        )
        return result


# ----------------------------------------------------------------------------
# Tool dispatch
# ----------------------------------------------------------------------------

class _ConcludeRequested(Exception):
    """Raised when the LLM emits ``action=conclude``. Caught by the loop."""


def _dispatch_action(
    action: AgentAction,
    state: AgentState,
    connection: CosmosConnection,
    collection: str,
    agent: ArgusAgent,
) -> None:
    if action.action == "schema_sample":
        n = int(action.action_input.get("sample_size", agent.config.sample_size))
        schema_result = schema_sample(connection, collection, sample_size=n)
        state.schema = schema_result
        state.documents_sampled = max(state.documents_sampled, schema_result.documents_sampled)
        state.last_observation = (
            f"schema_sample: docs_sampled={schema_result.documents_sampled}, "
            f"distinct_paths={len(schema_result.fields)}, "
            f"truncated_paths={len(schema_result.truncated_paths)}"
        )

    elif action.action == "run_query":
        filt = action.action_input.get("filter") or {}
        if not isinstance(filt, dict):
            raise TypeError("run_query.filter must be a dict")
        limit = int(action.action_input.get("limit", 50))
        query_result = run_query(connection, collection, filter=filt, limit=limit)
        state.queries_run.append(filt)
        state.fields_investigated.update(_filter_fields(filt))
        sample_keys = sorted(query_result.documents[0].keys()) if query_result.documents else []
        state.last_observation = (
            f"run_query: matched_count={query_result.matched_count}, "
            f"returned={query_result.returned_count}, "
            f"truncated={query_result.truncated}, sample_doc_keys={sample_keys}"
        )

    elif action.action == "get_stats":
        field_path = action.action_input.get("field")
        op = action.action_input.get("operation")
        if not isinstance(field_path, str) or op not in {"count", "min", "max", "avg", "distinct"}:
            raise ValueError(
                f"get_stats requires field:str and operation in count|min|max|avg|distinct "
                f"(got field={field_path!r}, op={op!r})"
            )
        stats_result = get_stats(connection, collection, field_path, op)
        state.fields_investigated.add(field_path)
        repr_str = repr(stats_result.result)
        if len(repr_str) > 300:
            repr_str = repr_str[:300] + "…"
        state.last_observation = f"get_stats({field_path}, {op}): {repr_str}"

    elif action.action == "write_finding":
        _commit_finding(action, state, agent)

    elif action.action == "conclude":
        raise _ConcludeRequested

    else:  # pragma: no cover — Literal exhausts at type level
        raise ValueError(f"unknown action: {action.action!r}")


def _commit_finding(action: AgentAction, state: AgentState, agent: ArgusAgent) -> None:
    args = action.action_input
    affected = int(args.get("affected_count", 0))
    affected_pct_raw = args.get("affected_pct")
    if affected_pct_raw is None:
        affected_pct = (affected / state.documents_sampled) if state.documents_sampled else 0.0
    else:
        affected_pct = float(affected_pct_raw)
    affected_pct = max(0.0, min(1.0, affected_pct))

    severity_raw = str(args.get("severity", "medium")).lower()
    try:
        severity = FindingSeverity(severity_raw)
    except ValueError:
        severity = FindingSeverity.MEDIUM

    candidate = Finding(
        field=str(args.get("field", "")),
        category=str(args.get("category", "unknown")),
        severity=severity,
        description=str(args.get("description", "")),
        hypothesis=str(args.get("hypothesis", "")),
        evidence_query=str(args.get("evidence_query", "")),
        affected_count=affected,
        affected_pct=affected_pct,
        sample_values=list(args.get("sample_values", [])[:5]),
        confirmed=bool(args.get("confirmed", True)),
    )

    verdict = agent._evaluate_finding(candidate, state)
    if verdict and verdict.verdict == EvaluationVerdict.FAIL:
        policy = agent.config.evaluation.rejected_finding_policy
        if policy == "drop":
            state.last_observation = f"write_finding REJECTED ({verdict.reason}); dropped"
            return
        if policy == "log_only":
            state.dismissed_findings.append(candidate)
            state.last_observation = f"write_finding REJECTED ({verdict.reason}); dismissed"
            state.last_critique = verdict.critique or verdict.reason
            return
        if policy == "demote_severity":
            demoted = _demote(severity)
            candidate = candidate.model_copy(update={"severity": demoted})
            state.last_observation = (
                f"write_finding severity demoted {severity.value}→{demoted.value} ({verdict.reason})"
            )

    fid: UUID = state.findings.write(
        field=candidate.field,
        category=candidate.category,
        severity=candidate.severity,
        description=candidate.description,
        hypothesis=candidate.hypothesis,
        evidence_query=candidate.evidence_query,
        affected_count=candidate.affected_count,
        affected_pct=candidate.affected_pct,
        sample_values=candidate.sample_values,
        confirmed=candidate.confirmed,
    )
    state.fields_concluded.add(candidate.field)
    if state.last_observation is None or "REJECTED" not in state.last_observation:
        state.last_observation = (
            f"write_finding committed id={fid} field={candidate.field} category={candidate.category}"
        )


def _demote(s: FindingSeverity) -> FindingSeverity:
    order = [FindingSeverity.LOW, FindingSeverity.MEDIUM, FindingSeverity.HIGH, FindingSeverity.CRITICAL]
    idx = order.index(s)
    return order[max(0, idx - 1)]


def _filter_fields(filt: dict[str, Any], prefix: str = "") -> set[str]:
    """Best-effort: collect field paths referenced in a Mongo filter."""
    out: set[str] = set()
    for k, v in filt.items():
        if k.startswith("$"):
            if isinstance(v, list):
                for sub in v:
                    if isinstance(sub, dict):
                        out.update(_filter_fields(sub, prefix))
            continue
        path = f"{prefix}.{k}" if prefix else k
        out.add(path)
        if isinstance(v, dict):
            # Nested operator dict ({"$gt": 10}) — keep the parent path only.
            inner_field_keys = [ik for ik in v if not ik.startswith("$")]
            if inner_field_keys:
                out.update(_filter_fields(v, path))
    return out


def _safe_collection_size(connection: CosmosConnection, collection: str) -> int:
    try:
        return int(connection.collection(collection).estimated_document_count())
    except Exception:  # noqa: BLE001 — Cosmos may throttle estimated_count; the field is informational
        return 0


def _build_report(
    state: AgentState,
    started: float,
    run_evaluation: EvaluationResult | None,
) -> AuditReport:
    duration = time.time() - started
    quality_score = run_evaluation.score if run_evaluation else None
    return AuditReport(
        collection=state.collection,
        database=state.database,
        cosmos_account=state.cosmos_account,
        duration_seconds=duration,
        documents_sampled=state.documents_sampled,
        collection_size=state.collection_size,
        findings=state.findings.all(),
        run_trace=list(state.history),
        summary=_synthesize_summary(state),
        total_input_tokens=state.total_usage.input_tokens,
        total_output_tokens=state.total_usage.output_tokens,
        evaluation_records=list(state.evaluation_records),
        dismissed_findings=list(state.dismissed_findings),
        run_evaluation=run_evaluation,
        overall_quality_score=quality_score,
    )


def _synthesize_summary(state: AgentState) -> str:
    findings = state.findings.all()
    by_sev: dict[str, int] = {}
    for f in findings:
        by_sev[f.severity.value] = by_sev.get(f.severity.value, 0) + 1
    parts = [
        f"Sampled {state.documents_sampled} docs across {len(state.field_paths)} field paths.",
        f"Iterations used: {state.iteration}/{state.iteration_budget}.",
        f"Findings: {len(findings)} committed, {len(state.dismissed_findings)} dismissed.",
    ]
    if by_sev:
        parts.append("Severity breakdown: " + ", ".join(f"{k}={v}" for k, v in sorted(by_sev.items())))
    return " ".join(parts)
