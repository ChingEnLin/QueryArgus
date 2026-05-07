"""Tests for ReportStore — uses a fake psycopg2 connection so no real DB is needed.

We test the *contract* of the store (round-tripping AuditReport via raw_report,
correct child-row inserts, correct row-mapping in list_reports / get_previous)
without exercising real SQL. Real-Postgres integration testing is a separate
manual step (see PLAN.md).
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest

from queryargus.models.finding import Finding, FindingSeverity
from queryargus.models.report import AuditReport
from queryargus.storage import ReportStore


class _FakeCursor:
    """Captures executes; returns scripted rows from queue."""

    def __init__(self, rows: list[Any] | None = None) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self._rows: list[Any] = list(rows or [])

    def execute(self, sql: str, args: Any = ()) -> None:
        self.executed.append((sql, tuple(args) if isinstance(args, list | tuple) else (args,)))

    def fetchone(self) -> Any | None:
        return self._rows.pop(0) if self._rows else None

    def fetchall(self) -> list[Any]:
        rows, self._rows = self._rows, []
        return rows

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.committed = False
        self.closed = False

    def cursor(self, cursor_factory: Any = None) -> _FakeCursor:
        return self._cursor

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.committed = True


@pytest.fixture
def store_with_cursor(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    cursor = _FakeCursor()
    conn = _FakeConnection(cursor)

    def fake_connect(dsn: str, **_: Any) -> _FakeConnection:
        return conn

    monkeypatch.setattr("queryargus.storage.postgres.psycopg2.connect", fake_connect)
    return ReportStore(dsn="postgresql://fake/fake"), cursor


def _make_report() -> AuditReport:
    return AuditReport(
        collection="users", database="db", cosmos_account="acct",
        duration_seconds=2.5, documents_sampled=100, collection_size=1000,
        findings=[
            Finding(
                field="email", category="null_rate", severity=FindingSeverity.HIGH,
                description="Many nulls in email",
                hypothesis="optional-by-design or broken signup",
                evidence_query='{"email": null}',
                affected_count=30, affected_pct=0.30,
                sample_values=[None, "a@x.com"],
            ),
        ],
        total_input_tokens=1000,
        total_output_tokens=200,
    )


def test_save_writes_report_then_findings(store_with_cursor) -> None:  # type: ignore[no-untyped-def]
    store, cursor = store_with_cursor
    report = _make_report()
    store.save(report)

    sqls = [sql for sql, _ in cursor.executed]
    # 1 INSERT into argus_reports, 3 DELETEs (children), 1 INSERT into argus_findings.
    assert any("INSERT INTO argus_reports" in s for s in sqls)
    assert any("DELETE FROM argus_findings" in s for s in sqls)
    assert any("INSERT INTO argus_findings" in s for s in sqls)


def test_save_then_get_round_trips_via_raw_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """get() reconstructs an AuditReport from raw_report JSONB stored at save time."""
    report = _make_report()
    raw = json.loads(report.model_dump_json())
    cursor = _FakeCursor(rows=[{"raw_report": raw}])
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(
        "queryargus.storage.postgres.psycopg2.connect", lambda *_a, **_kw: conn
    )

    store = ReportStore(dsn="postgresql://fake/fake")
    loaded = store.get(report.id)
    assert loaded is not None
    assert loaded.collection == report.collection
    assert loaded.findings[0].category == "null_rate"
    assert loaded.total_input_tokens == 1000


def test_list_reports_maps_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    rid = uuid4()
    cursor = _FakeCursor(
        rows=[
            {
                "id": str(rid),
                "collection": "users",
                "database": "db",
                "cosmos_account": "acct",
                "run_at": __import__("datetime").datetime(2026, 5, 1, 12, 0, 0),
                "overall_quality_score": 0.8,
                "run_eval_verdict": "warn",
                "total_input_tokens": 100,
                "total_output_tokens": 20,
                "findings_count": 4,
            }
        ]
    )
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(
        "queryargus.storage.postgres.psycopg2.connect", lambda *_a, **_kw: conn
    )

    store = ReportStore(dsn="postgresql://fake/fake")
    rows = store.list_reports(collection="users", database="db", limit=5)
    assert len(rows) == 1
    assert rows[0].collection == "users"
    assert rows[0].findings_count == 4
    assert rows[0].total_input_tokens == 100


def test_get_previous_returns_none_when_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor(rows=[])  # no row found
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(
        "queryargus.storage.postgres.psycopg2.connect", lambda *_a, **_kw: conn
    )

    store = ReportStore(dsn="postgresql://fake/fake")
    assert store.get_previous(collection="x", database="y") is None
