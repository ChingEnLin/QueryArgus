"""AuditReport — the final structured output of an audit run."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from queryargus.models.action import AgentAction
from queryargus.models.evaluation import EvaluationRecord, EvaluationResult
from queryargus.models.finding import Finding


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ModelCost(BaseModel):
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    usd: float = Field(ge=0.0)


class RunCost(BaseModel):
    usd_total: float = Field(default=0.0, ge=0.0)
    by_model: list[ModelCost] = Field(default_factory=list)
    pricing_version: str = "2026-05"


class AuditReport(BaseModel):
    """The structured output of one audit run. See spec §6.2."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    collection: str = Field(min_length=1)
    database: str = Field(min_length=1)
    cosmos_account: str = Field(min_length=1)
    run_at: datetime = Field(default_factory=_utcnow)
    duration_seconds: float = Field(ge=0.0)
    documents_sampled: int = Field(ge=0)
    collection_size: int = Field(ge=0)

    findings: list[Finding] = Field(default_factory=list)
    run_trace: list[AgentAction] = Field(default_factory=list)
    summary: str = ""

    # Token usage across all LLM calls in this run (zero when the loop ran with
    # ScriptedLLMClient or another provider that does not report usage).
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    cost: RunCost | None = None

    # Evaluation outputs (populated by the evaluation layer — see spec §5)
    evaluation_records: list[EvaluationRecord] = Field(default_factory=list)
    dismissed_findings: list[Finding] = Field(default_factory=list)
    run_evaluation: EvaluationResult | None = None
    overall_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)

    # Diff against previous run (populated when storage is configured)
    previous_run_id: UUID | None = None
    new_findings: list[Finding] = Field(default_factory=list)
    resolved_findings: list[Finding] = Field(default_factory=list)
    regressed_fields: list[str] = Field(default_factory=list)

    def diff_against(self, previous: AuditReport, *, regression_factor: float = 1.5) -> AuditReport:
        """Return a copy with diff fields populated against ``previous``.

        - ``new_findings``: findings whose ``(field, category)`` did not appear before.
        - ``resolved_findings``: previous findings whose ``(field, category)`` is gone.
        - ``regressed_fields``: fields whose severity rose OR whose ``affected_pct``
          climbed by ``regression_factor`` (default 1.5×). Deduplicated.

        Pure function — does not touch storage. Idempotent.
        """
        prev_index = {(f.field, f.category): f for f in previous.findings}
        curr_index = {(f.field, f.category): f for f in self.findings}

        new = [f for k, f in curr_index.items() if k not in prev_index]
        resolved = [f for k, f in prev_index.items() if k not in curr_index]

        sev_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        regressed: list[str] = []
        for key, cur in curr_index.items():
            prev = prev_index.get(key)
            if prev is None:
                continue
            severity_worse = sev_rank.get(cur.severity.value, 0) > sev_rank.get(prev.severity.value, 0)
            pct_worse = (
                prev.affected_pct > 0 and cur.affected_pct > prev.affected_pct * regression_factor
            )
            if severity_worse or pct_worse:
                regressed.append(cur.field)

        return self.model_copy(update={
            "previous_run_id": previous.id,
            "new_findings": new,
            "resolved_findings": resolved,
            "regressed_fields": sorted(set(regressed)),
        })
