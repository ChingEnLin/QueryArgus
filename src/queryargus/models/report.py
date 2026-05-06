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
