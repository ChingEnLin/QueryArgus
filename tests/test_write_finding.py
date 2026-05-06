"""Tests for FindingsCollector idempotency."""

from __future__ import annotations

from typing import Any

from queryargus.models.finding import FindingSeverity
from queryargus.tools.write_finding import FindingsCollector


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "field": "user.age",
        "category": "outlier_value",
        "severity": FindingSeverity.HIGH,
        "description": "Outlier values present",
        "hypothesis": "Sentinel like 9999",
        "evidence_query": "{age: {$gt: 150}}",
        "affected_count": 3,
        "affected_pct": 0.03,
        "sample_values": [9999],
    }
    base.update(overrides)
    return base


def test_first_write_creates_finding() -> None:
    c = FindingsCollector()
    fid = c.write(**_kwargs())
    assert fid is not None
    assert len(c) == 1
    assert ("user.age", "outlier_value") in c


def test_second_write_same_key_updates_in_place_and_keeps_id() -> None:
    c = FindingsCollector()
    fid1 = c.write(**_kwargs())
    fid2 = c.write(**_kwargs(severity=FindingSeverity.CRITICAL, affected_count=99))
    assert fid1 == fid2
    assert len(c) == 1
    only = c.all()[0]
    assert only.severity == FindingSeverity.CRITICAL
    assert only.affected_count == 99


def test_different_category_creates_separate_finding() -> None:
    c = FindingsCollector()
    c.write(**_kwargs())
    c.write(**_kwargs(category="null_rate"))
    assert len(c) == 2
