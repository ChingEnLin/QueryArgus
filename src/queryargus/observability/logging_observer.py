"""StructuredLogObserver — one JSON-ready log record per agent event.

Calls ``logger.info("event_name", extra={...})`` for every hook. Does not
configure handlers — callers attach their own (the ``JsonFormatter`` shipped
here is a convenience). Payload discipline: never log raw tool arguments,
raw tool results, document samples, or finding descriptions. Logs flow into
the host's pipeline; Cosmos data must not leak there.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from queryargus.llm.client import TokenUsage
from queryargus.models.action import AgentAction
from queryargus.models.finding import Finding
from queryargus.models.report import AuditReport


def _ts() -> str:
    return datetime.now(UTC).isoformat()


class StructuredLogObserver:
    """Render every observer hook as a structured log record."""

    def __init__(
        self,
        logger_name: str = "queryargus.run",
        level: int = logging.INFO,
    ) -> None:
        self._logger = logging.getLogger(logger_name)
        self._level = level
        self._run_id: str | None = None

    def _log(self, event: str, /, **extras: object) -> None:
        payload: dict[str, object] = {
            "event": event,
            "run_id": self._run_id,
            "ts": _ts(),
            **extras,
        }
        self._logger.log(self._level, event, extra=payload)

    def on_run_start(self, *, run_id: UUID, collection: str) -> None:
        self._run_id = str(run_id)
        self._log("run_start", collection=collection)

    def on_iteration_start(self, *, iter: int) -> None:
        self._log("iteration_start", iter=iter)

    def on_llm_call(
        self,
        *,
        purpose: Literal["propose_action", "self_eval", "judge"],
        model: str,
        usage: TokenUsage,
        latency_ms: int,
    ) -> None:
        self._log(
            "llm_call",
            purpose=purpose,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
        )

    def on_tool_call(
        self,
        *,
        name: str,
        args_summary: str,
        ok: bool,
        latency_ms: int,
        error: str | None,
    ) -> None:
        self._log(
            "tool_call",
            tool=name,
            args_summary=args_summary,
            ok=ok,
            latency_ms=latency_ms,
            error=error,
        )

    def on_action(self, *, action: AgentAction) -> None:
        self._log("action", action=action.action, confidence=action.confidence)

    def on_finding(self, *, finding: Finding) -> None:
        self._log(
            "finding",
            field=finding.field,
            category=finding.category,
            severity=str(finding.severity),
        )

    def on_eval(
        self,
        *,
        target: Literal["action", "finding", "run"],
        verdict: str,
        score: float,
        evaluator: str,
    ) -> None:
        self._log(
            "eval",
            target=target,
            verdict=verdict,
            score=score,
            evaluator=evaluator,
        )

    def on_run_complete(self, *, report: AuditReport) -> None:
        self._log(
            "run_complete",
            findings_count=len(report.findings),
            usd_total=(report.cost.usd_total if report.cost is not None else None),
            total_input_tokens=report.total_input_tokens,
            total_output_tokens=report.total_output_tokens,
            duration_ms=int(report.duration_seconds * 1000),
        )


_BUILTIN_LOGRECORD_KEYS = set(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Render a LogRecord plus any extra attributes as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k in _BUILTIN_LOGRECORD_KEYS:
                continue
            base[k] = v
        return json.dumps(base, default=str)
