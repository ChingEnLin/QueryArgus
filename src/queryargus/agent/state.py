"""AgentState — the running record of an investigation, mutated through the loop.

The state is what the planner shows the LLM (via :meth:`summarize`) and what the
evaluators read to make their decisions. It is *not* the audit report — that is
materialized at the end of the run from the state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from queryargus.llm.client import TokenUsage
from queryargus.models.action import AgentAction
from queryargus.models.evaluation import EvaluationRecord
from queryargus.models.finding import Finding
from queryargus.models.history import HistoricalContext
from queryargus.tools.schema_sample import FieldStats, SchemaSampleResult
from queryargus.tools.write_finding import FindingsCollector

_QUERY_TAIL = 20
_ACTION_TAIL = 10
_FIELD_DISPLAY_LIMIT = 60


@dataclass
class AgentState:
    """Mutable per-run state, passed to planner and evaluators each iteration."""

    collection: str
    database: str
    cosmos_account: str
    iteration_budget: int

    iteration: int = 0
    documents_sampled: int = 0
    collection_size: int = 0

    schema: SchemaSampleResult | None = None
    queries_run: list[dict[str, Any]] = field(default_factory=list)
    fields_investigated: set[str] = field(default_factory=set)
    fields_concluded: set[str] = field(default_factory=set)

    history: list[AgentAction] = field(default_factory=list)
    last_observation: str | None = None
    last_critique: str | None = None

    findings: FindingsCollector = field(default_factory=FindingsCollector)
    dismissed_findings: list[Finding] = field(default_factory=list)
    evaluation_records: list[EvaluationRecord] = field(default_factory=list)

    total_usage: TokenUsage = field(default_factory=TokenUsage)
    usage_per_iteration: list[TokenUsage] = field(default_factory=list)

    historical_context: HistoricalContext | None = None

    @property
    def remaining_budget(self) -> int:
        return max(0, self.iteration_budget - self.iteration)

    @property
    def field_paths(self) -> list[str]:
        return [f.path for f in self.schema.fields] if self.schema else []

    def field_stats(self, path: str) -> FieldStats | None:
        if not self.schema:
            return None
        for f in self.schema.fields:
            if f.path == path:
                return f
        return None

    def has_run_query(self, filter_: dict[str, Any]) -> bool:
        return filter_ in self.queries_run

    def summarize(self) -> str:
        """Compact, LLM-friendly snapshot of the investigation.

        Structured as a *stable prefix* followed by a *volatile trailer* so the
        request is cache-friendly. The prefix — collection identity, the sampled
        SCHEMA, and the HISTORICAL CONTEXT — is byte-identical on every turn once
        the schema is fixed, letting Gemini's implicit cache (and cachelens's
        prefix detector) anchor on it. Everything that mutates per iteration —
        the iteration counter, queries run, recent actions, findings, critique,
        last observation — lives in the trailer. Nothing volatile (notably the
        running token tally, which changed every call) is allowed in the prefix.
        """
        return "\n".join([*self._stable_prefix_lines(), "", *self._volatile_trailer_lines()])

    def _stable_prefix_lines(self) -> list[str]:
        """Fixed-for-the-run context. Must not contain anything that changes per turn."""
        lines: list[str] = ["=== COLLECTION UNDER AUDIT (fixed context) ==="]
        lines.append(f"collection: {self.collection}  database: {self.database}  account: {self.cosmos_account}")
        lines.append(f"documents_sampled: {self.documents_sampled}  collection_size: {self.collection_size}")

        if self.schema is None:
            lines.append("\nSCHEMA: not yet sampled — call schema_sample first.")
        else:
            lines.append(f"\nSCHEMA ({len(self.schema.fields)} field paths):")
            for fs in self.schema.fields[:_FIELD_DISPLAY_LIMIT]:
                types = ",".join(f"{t}={c}" for t, c in sorted(fs.types.items()))
                lines.append(
                    f"  {fs.path}: present={fs.present_count} missing={fs.missing_count} "
                    f"null_rate={fs.null_rate:.3f} card={fs.cardinality}"
                    f"{'+' if fs.cardinality_capped else ''} types={types}"
                )
            if len(self.schema.fields) > _FIELD_DISPLAY_LIMIT:
                lines.append(f"  ... ({len(self.schema.fields) - _FIELD_DISPLAY_LIMIT} more not shown)")

        if self.historical_context is not None and not self.historical_context.is_empty:
            lines.append("")
            lines.append(self.historical_context.render())

        return lines

    def _volatile_trailer_lines(self) -> list[str]:
        """Per-iteration progress. Everything here is expected to change between turns."""
        lines: list[str] = [f"=== INVESTIGATION PROGRESS (iteration {self.iteration}/{self.iteration_budget}) ==="]

        if self.fields_investigated:
            lines.append(f"INVESTIGATED FIELDS ({len(self.fields_investigated)}): "
                         f"{', '.join(sorted(self.fields_investigated))}")
        if self.fields_concluded:
            lines.append(f"CONCLUDED FIELDS ({len(self.fields_concluded)}): "
                         f"{', '.join(sorted(self.fields_concluded))}")

        if self.queries_run:
            lines.append(f"\nQUERIES RUN ({len(self.queries_run)}):")
            for q in self.queries_run[-_QUERY_TAIL:]:
                lines.append(f"  {json.dumps(q, default=str)}")

        if self.history:
            lines.append(f"\nRECENT ACTIONS ({len(self.history)} total, last {_ACTION_TAIL}):")
            for a in self.history[-_ACTION_TAIL:]:
                lines.append(f"  - {a.action} (conf={a.confidence:.2f}): {a.reasoning[:140]}")

        committed = self.findings.all()
        if committed:
            lines.append(f"\nFINDINGS COMMITTED ({len(committed)}):")
            for f_ in committed:
                lines.append(
                    f"  - [{f_.severity.value}] {f_.field} / {f_.category}: "
                    f"{f_.description[:140]} (affected={f_.affected_count}, pct={f_.affected_pct:.3f})"
                )

        if self.dismissed_findings:
            lines.append(f"\nFINDINGS DISMISSED BY EVALUATOR ({len(self.dismissed_findings)}):")
            for f_ in self.dismissed_findings[-5:]:
                lines.append(f"  - {f_.field} / {f_.category}: {f_.description[:120]}")

        if self.last_critique:
            lines.append(f"\nEVALUATOR CRITIQUE (apply on next iteration): {self.last_critique}")

        if self.last_observation:
            lines.append(f"\nLAST OBSERVATION: {self.last_observation[:600]}")

        return lines
