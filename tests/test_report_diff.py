"""Tests for AuditReport.diff_against."""

from __future__ import annotations

from queryargus.models.finding import Finding, FindingSeverity
from queryargus.models.report import AuditReport


def _f(field: str, category: str, severity: FindingSeverity, pct: float, count: int = 10) -> Finding:
    return Finding(
        field=field, category=category, severity=severity,
        description=f"{field} has {category}",
        hypothesis="x", evidence_query="{}",
        affected_count=count, affected_pct=pct,
    )


def _report(*findings: Finding) -> AuditReport:
    return AuditReport(
        collection="users", database="db", cosmos_account="acct",
        duration_seconds=1.0, documents_sampled=100, collection_size=1000,
        findings=list(findings),
    )


def test_diff_marks_new_findings() -> None:
    prev = _report(_f("age", "outlier_value", FindingSeverity.MEDIUM, 0.05))
    curr = _report(
        _f("age", "outlier_value", FindingSeverity.MEDIUM, 0.05),
        _f("email", "null_rate", FindingSeverity.HIGH, 0.30),
    )
    diffed = curr.diff_against(prev)
    assert {f.field for f in diffed.new_findings} == {"email"}
    assert diffed.previous_run_id == prev.id


def test_diff_marks_resolved_findings() -> None:
    prev = _report(
        _f("age", "outlier_value", FindingSeverity.MEDIUM, 0.05),
        _f("email", "null_rate", FindingSeverity.HIGH, 0.30),
    )
    curr = _report(_f("age", "outlier_value", FindingSeverity.MEDIUM, 0.05))
    diffed = curr.diff_against(prev)
    assert {f.field for f in diffed.resolved_findings} == {"email"}


def test_diff_severity_regression() -> None:
    prev = _report(_f("age", "outlier_value", FindingSeverity.MEDIUM, 0.05))
    curr = _report(_f("age", "outlier_value", FindingSeverity.HIGH, 0.05))
    diffed = curr.diff_against(prev)
    assert "age" in diffed.regressed_fields


def test_diff_pct_regression_above_factor() -> None:
    prev = _report(_f("age", "outlier_value", FindingSeverity.MEDIUM, 0.05))
    curr = _report(_f("age", "outlier_value", FindingSeverity.MEDIUM, 0.20))  # 4x
    diffed = curr.diff_against(prev)
    assert "age" in diffed.regressed_fields


def test_diff_minor_pct_change_not_a_regression() -> None:
    prev = _report(_f("age", "outlier_value", FindingSeverity.MEDIUM, 0.05))
    curr = _report(_f("age", "outlier_value", FindingSeverity.MEDIUM, 0.06))  # 1.2x — under default 1.5x
    diffed = curr.diff_against(prev)
    assert "age" not in diffed.regressed_fields


def test_diff_is_pure_does_not_mutate_self() -> None:
    prev = _report(_f("age", "outlier_value", FindingSeverity.MEDIUM, 0.05))
    curr = _report(
        _f("age", "outlier_value", FindingSeverity.MEDIUM, 0.05),
        _f("email", "null_rate", FindingSeverity.HIGH, 0.30),
    )
    _ = curr.diff_against(prev)
    # Original is untouched.
    assert curr.new_findings == []
    assert curr.previous_run_id is None
