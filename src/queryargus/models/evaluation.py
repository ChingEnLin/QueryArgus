"""Evaluation models — verdicts, results, and per-decision records."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class EvaluationVerdict(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class EvaluationResult(BaseModel):
    """The verdict from a single evaluator call."""

    model_config = ConfigDict(extra="forbid")

    verdict: EvaluationVerdict
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    critique: str | None = None
    evaluated_by: str = Field(min_length=1, description="e.g. 'rules', 'self:gemini-pro', 'judge:gpt-4o'.")


class EvaluationRecord(BaseModel):
    """One row in the report's audit trail of evaluation decisions."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    gate: Literal["action", "finding", "run"]
    evaluated_by: str = Field(min_length=1)
    verdict: EvaluationVerdict
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    critique: str | None = None
    target_id: UUID | None = None
    iteration: int | None = None
    timestamp: datetime = Field(default_factory=_utcnow)
