from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from queryargus.llm.client import TokenUsage
from queryargus.models.finding import Finding, FindingSeverity
from queryargus.models.report import AuditReport
from queryargus.observability.logging_observer import (
    JsonFormatter,
    StructuredLogObserver,
)


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    builtins = set(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {
        "message",
        "asctime",
    }
    return {k: v for k, v in record.__dict__.items() if k not in builtins}


def _fresh_report(**overrides: Any) -> AuditReport:
    base: dict[str, Any] = {
        "collection": "c",
        "database": "d",
        "cosmos_account": "a",
        "duration_seconds": 0.0,
        "documents_sampled": 0,
        "collection_size": 0,
    }
    base.update(overrides)
    return AuditReport(**base)


def test_run_start_emits_event(caplog: Any) -> None:
    obs = StructuredLogObserver()
    rid = uuid4()
    with caplog.at_level(logging.INFO, logger="queryargus.run"):
        obs.on_run_start(run_id=rid, collection="orders")
    assert len(caplog.records) == 1
    extras = _extras(caplog.records[0])
    assert extras["event"] == "run_start"
    assert extras["collection"] == "orders"
    assert extras["run_id"] == str(rid)
    assert "ts" in extras


def test_llm_call_payload(caplog: Any) -> None:
    obs = StructuredLogObserver()
    obs.on_run_start(run_id=uuid4(), collection="c")
    with caplog.at_level(logging.INFO, logger="queryargus.run"):
        obs.on_llm_call(
            purpose="propose_action",
            model="gemini-2.5-flash",
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            latency_ms=42,
        )
    extras = _extras(caplog.records[-1])
    assert extras["event"] == "llm_call"
    assert extras["purpose"] == "propose_action"
    assert extras["model"] == "gemini-2.5-flash"
    assert extras["input_tokens"] == 100
    assert extras["output_tokens"] == 50
    assert extras["latency_ms"] == 42


def test_tool_call_payload_does_not_leak_args(caplog: Any) -> None:
    obs = StructuredLogObserver()
    obs.on_run_start(run_id=uuid4(), collection="c")
    secret = "sk-extremely-secret-token-do-not-log"
    with caplog.at_level(logging.INFO, logger="queryargus.run"):
        obs.on_tool_call(
            name="run_query",
            args_summary="filter=<dict 4 keys>, limit=100",
            ok=True,
            latency_ms=12,
            error=None,
        )
    for record in caplog.records:
        assert secret not in record.getMessage()
        for v in _extras(record).values():
            assert secret not in str(v)


def test_finding_payload_excludes_sample_data(caplog: Any) -> None:
    obs = StructuredLogObserver()
    obs.on_run_start(run_id=uuid4(), collection="c")
    finding = Finding(
        field="user.email",
        category="null_rate",
        severity=FindingSeverity.HIGH,
        description="sensitive details that should not be logged",
        hypothesis="hyp",
        evidence_query="db.users.count({email: null})",
        affected_count=1,
        affected_pct=0.01,
        sample_values=["nope@example.com"],
    )
    with caplog.at_level(logging.INFO, logger="queryargus.run"):
        obs.on_finding(finding=finding)
    extras = _extras(caplog.records[-1])
    assert extras["event"] == "finding"
    assert extras["field"] == "user.email"
    assert extras["category"] == "null_rate"
    assert extras["severity"] == "high"
    for forbidden in ("description", "hypothesis", "evidence_query", "sample_values"):
        assert forbidden not in extras
    for v in extras.values():
        assert "sensitive details" not in str(v)
        assert "nope@example.com" not in str(v)


def test_run_complete_payload(caplog: Any) -> None:
    obs = StructuredLogObserver()
    obs.on_run_start(run_id=uuid4(), collection="c")
    report = _fresh_report(
        duration_seconds=1.5,
        total_input_tokens=1000,
        total_output_tokens=500,
    )
    with caplog.at_level(logging.INFO, logger="queryargus.run"):
        obs.on_run_complete(report=report)
    extras = _extras(caplog.records[-1])
    assert extras["event"] == "run_complete"
    assert extras["findings_count"] == 0
    assert extras["total_input_tokens"] == 1000
    assert extras["total_output_tokens"] == 500
    assert extras["duration_ms"] == 1500


def test_json_formatter_produces_valid_json() -> None:
    fmt = JsonFormatter()
    rec = logging.LogRecord(
        name="queryargus.run",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="event",
        args=(),
        exc_info=None,
    )
    rec.event = "iteration_start"  # type: ignore[attr-defined]
    rec.run_id = "abc"  # type: ignore[attr-defined]
    rec.iter = 1  # type: ignore[attr-defined]
    payload = json.loads(fmt.format(rec))
    assert payload["event"] == "iteration_start"
    assert payload["run_id"] == "abc"
    assert payload["iter"] == 1
    assert payload["level"] == "INFO"
