"""Finding model — a single confirmed data quality issue."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

UserLabel = Literal["tp", "fp"]
FindingStatus = Literal["published", "pending_review", "dropped"]


class FindingSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Finding(BaseModel):
    """A single confirmed data quality issue.

    `affected_pct` is a fraction in [0.0, 1.0] (so 0.05 = 5%), not a percentage —
    this composes with `ArgusConfig.null_rate_*_threshold` which use the same scale.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    field: str = Field(min_length=1, description="Dot-notation field path, e.g. 'user.profile.age'.")
    category: str = Field(min_length=1, description="e.g. 'null_rate', 'type_mismatch', 'outlier_value'.")
    severity: FindingSeverity
    description: str = Field(min_length=1)
    hypothesis: str
    evidence_query: str
    affected_count: int = Field(ge=0)
    affected_pct: float = Field(ge=0.0, le=1.0)
    sample_values: list[Any] = Field(default_factory=list, max_length=5)
    confirmed: bool = True
    # Optional post-hoc verdict from a human reviewer. Drives cross-run
    # learning via UserVerdictHistory; never set by the agent itself.
    user_label: UserLabel | None = None
    # Arm B — self-assessment emitted by the agent at write_finding time.
    # ``confidence`` is sourced from AgentAction.confidence; ``confidence_reason``
    # mirrors AgentAction.reasoning. Both feed the EscalationFindingEvaluator.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_reason: str | None = None
    # Arm B — lifecycle marker. ``pending_review`` findings are persisted but
    # excluded from the user-facing count until a human resolves them.
    status: FindingStatus = "published"
