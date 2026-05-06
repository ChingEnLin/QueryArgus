"""Tests for the Pydantic models."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from queryargus.models import (
    PROFILE_BALANCED,
    PROFILE_FAST,
    PROFILE_THOROUGH,
    AgentAction,
    ArgusConfig,
    AuditReport,
    EvaluationRecord,
    EvaluationResult,
    EvaluationVerdict,
    EvaluatorConfig,
    Finding,
    FindingSeverity,
)


def _valid_finding_kwargs() -> dict[str, Any]:
    return {
        "field": "user.age",
        "category": "outlier_value",
        "severity": FindingSeverity.HIGH,
        "description": "Some users have impossible ages.",
        "hypothesis": "Data entry error or sentinel value.",
        "evidence_query": "db.users.count({age: {$gt: 150}})",
        "affected_count": 3,
        "affected_pct": 0.03,
        "sample_values": [9999, 1234],
    }


def test_finding_defaults_id_and_confirmed() -> None:
    f = Finding(**_valid_finding_kwargs())
    assert f.id is not None
    assert f.confirmed is True


def test_finding_rejects_pct_over_one() -> None:
    kwargs = _valid_finding_kwargs() | {"affected_pct": 1.5}
    with pytest.raises(ValidationError):
        Finding(**kwargs)


def test_finding_rejects_extra_field() -> None:
    kwargs = _valid_finding_kwargs() | {"unexpected": "x"}
    with pytest.raises(ValidationError):
        Finding(**kwargs)


def test_agent_action_validates_confidence_range() -> None:
    AgentAction(reasoning="r", action="schema_sample", confidence=0.5)
    with pytest.raises(ValidationError):
        AgentAction(reasoning="r", action="schema_sample", confidence=1.5)


def test_agent_action_rejects_unknown_action_name() -> None:
    with pytest.raises(ValidationError):
        AgentAction(reasoning="r", action="invalid_action", confidence=0.5)  # type: ignore[arg-type]


def test_evaluation_result_round_trip() -> None:
    r = EvaluationResult(
        verdict=EvaluationVerdict.PASS,
        score=0.9,
        reason="looks good",
        evaluated_by="rules",
    )
    j = r.model_dump_json()
    restored = EvaluationResult.model_validate_json(j)
    assert restored == r


def test_evaluation_record_defaults_id_and_timestamp() -> None:
    rec = EvaluationRecord(
        gate="finding",
        evaluated_by="rules",
        verdict=EvaluationVerdict.WARN,
        score=0.6,
        reason="borderline",
    )
    assert rec.id is not None
    assert rec.timestamp is not None


def test_evaluator_config_profiles_distinct() -> None:
    assert PROFILE_FAST.run_evaluator == "rules"
    assert PROFILE_BALANCED.run_evaluator == "self"
    assert PROFILE_THOROUGH.run_evaluator == "judge"


def test_argus_config_defaults_to_fast_like_profile() -> None:
    cfg = ArgusConfig()
    assert cfg.evaluation == EvaluatorConfig()
    assert cfg.evaluation.action_evaluator == "rules"


def test_audit_report_round_trip() -> None:
    report = AuditReport(
        collection="users",
        database="testdb",
        cosmos_account="test-account",
        duration_seconds=1.23,
        documents_sampled=100,
        collection_size=1000,
    )
    restored = AuditReport.model_validate_json(report.model_dump_json())
    assert restored.id == report.id
    assert restored.findings == []
