"""Postgres-backed audit-report store.

Raw psycopg2 (no ORM) — matches the QueryPal sibling and the locked decision
in the project context. The store keeps the *full* AuditReport in
``argus_reports.raw_report`` (JSONB) so loads are a single deserialize, and
mirrors findings / dismissed_findings / evaluation_records into the
respective relational tables for indexed queries (UI listings, calibration
analytics, etc).

Schema is idempotent: every ``ReportStore`` construction can call
``init_schema()`` to apply ``CREATE TABLE IF NOT EXISTS`` from ``schema.sql``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import psycopg2
from psycopg2.extensions import connection as PGConnection
from psycopg2.extras import Json, RealDictCursor

from queryargus.models.evaluation import EvaluationRecord
from queryargus.models.finding import Finding
from queryargus.models.history import FindingHistory, HistoricalContext, empty_history
from queryargus.models.report import AuditReport

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@dataclass(frozen=True)
class ReportSummary:
    """Lightweight row for ``list_reports`` — avoids deserializing every full report."""

    id: UUID
    collection: str
    database: str
    cosmos_account: str
    run_at: datetime
    findings_count: int
    overall_quality_score: float | None
    run_eval_verdict: str | None
    total_input_tokens: int
    total_output_tokens: int


class ReportStore:
    """Persistence facade. All public methods open + close their own connection."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    # ---- schema ------------------------------------------------------------

    def init_schema(self) -> None:
        """Apply ``schema.sql`` (idempotent — safe to call repeatedly)."""
        sql = _SCHEMA_PATH.read_text()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql)

    # ---- writes ------------------------------------------------------------

    def save(self, report: AuditReport) -> UUID:
        """Persist a report and all its child rows in a single transaction."""
        report_dict = report.model_dump(mode="json")
        run_eval_verdict = report.run_evaluation.verdict.value if report.run_evaluation else None
        run_eval_by = report.run_evaluation.evaluated_by if report.run_evaluation else None

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO argus_reports (
                    id, collection, database, cosmos_account, run_at,
                    duration_secs, docs_sampled, collection_size, summary,
                    overall_quality_score, run_eval_verdict, run_eval_by,
                    total_input_tokens, total_output_tokens,
                    run_trace, raw_report
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s
                )
                ON CONFLICT (id) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    overall_quality_score = EXCLUDED.overall_quality_score,
                    run_eval_verdict = EXCLUDED.run_eval_verdict,
                    run_eval_by = EXCLUDED.run_eval_by,
                    total_input_tokens = EXCLUDED.total_input_tokens,
                    total_output_tokens = EXCLUDED.total_output_tokens,
                    run_trace = EXCLUDED.run_trace,
                    raw_report = EXCLUDED.raw_report
                """,
                (
                    str(report.id), report.collection, report.database, report.cosmos_account,
                    report.run_at,
                    report.duration_seconds, report.documents_sampled, report.collection_size,
                    report.summary,
                    report.overall_quality_score, run_eval_verdict, run_eval_by,
                    report.total_input_tokens, report.total_output_tokens,
                    Json(report_dict["run_trace"]), Json(report_dict),
                ),
            )

            # Replace child rows on update so re-saves stay consistent.
            cur.execute("DELETE FROM argus_findings WHERE report_id = %s", (str(report.id),))
            cur.execute("DELETE FROM argus_dismissed_findings WHERE report_id = %s", (str(report.id),))
            cur.execute("DELETE FROM argus_evaluation_records WHERE report_id = %s", (str(report.id),))

            for f in report.findings:
                _insert_finding(cur, str(report.id), f, dismissed=False, dismiss_meta=None)
            for f in report.dismissed_findings:
                meta = _dismiss_meta_for(report, f)
                _insert_finding(cur, str(report.id), f, dismissed=True, dismiss_meta=meta)
            for rec in report.evaluation_records:
                _insert_eval_record(cur, str(report.id), rec)

        logger.info(
            "saved report id=%s collection=%s findings=%d",
            report.id, report.collection, len(report.findings),
        )
        return report.id

    # ---- reads -------------------------------------------------------------

    def get(self, report_id: UUID) -> AuditReport | None:
        """Load and reconstruct a full ``AuditReport`` from ``raw_report``."""
        with self._connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT raw_report FROM argus_reports WHERE id = %s", (str(report_id),))
            row: Any = cur.fetchone()
        if row is None:
            return None
        raw: Any = row["raw_report"]
        if isinstance(raw, str):
            return AuditReport.model_validate_json(raw)
        return AuditReport.model_validate(raw)

    def list_reports(
        self,
        *,
        collection: str | None = None,
        database: str | None = None,
        limit: int = 10,
    ) -> list[ReportSummary]:
        clauses: list[str] = []
        args: list[Any] = []
        if collection is not None:
            clauses.append("collection = %s")
            args.append(collection)
        if database is not None:
            clauses.append("database = %s")
            args.append(database)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT
                r.id, r.collection, r.database, r.cosmos_account, r.run_at,
                r.overall_quality_score, r.run_eval_verdict,
                r.total_input_tokens, r.total_output_tokens,
                (SELECT COUNT(*) FROM argus_findings f WHERE f.report_id = r.id) AS findings_count
            FROM argus_reports r
            {where}
            ORDER BY r.run_at DESC
            LIMIT %s
        """
        args.append(limit)
        with self._connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, args)
            rows = cur.fetchall()
        return [
            ReportSummary(
                id=UUID(str(row["id"])),
                collection=row["collection"],
                database=row["database"],
                cosmos_account=row["cosmos_account"],
                run_at=row["run_at"],
                findings_count=int(row["findings_count"] or 0),
                overall_quality_score=row["overall_quality_score"],
                run_eval_verdict=row["run_eval_verdict"],
                total_input_tokens=int(row["total_input_tokens"] or 0),
                total_output_tokens=int(row["total_output_tokens"] or 0),
            )
            for row in rows
        ]

    def load_history(
        self,
        *,
        collection: str,
        database: str,
        limit: int = 5,
    ) -> HistoricalContext:
        """Aggregate the last ``limit`` runs into a ``HistoricalContext``.

        Returns an empty context when no prior runs exist for the
        ``(collection, database)`` pair. The agent reads this at run start to
        avoid rediscovering known issues from scratch.
        """
        with self._connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, run_at FROM argus_reports
                WHERE collection = %s AND database = %s
                ORDER BY run_at DESC LIMIT %s
                """,
                (collection, database, limit),
            )
            runs = cur.fetchall()
            if not runs:
                return empty_history()

            run_ids = [str(r["id"]) for r in runs]
            last_run_at = runs[0]["run_at"]

            cur.execute(
                """
                SELECT f.field, f.category, f.severity, f.affected_pct, r.run_at
                FROM argus_findings f
                JOIN argus_reports r ON f.report_id = r.id
                WHERE f.report_id::text = ANY(%s)
                ORDER BY r.run_at DESC
                """,
                (run_ids,),
            )
            finding_rows = cur.fetchall()

            cur.execute(
                """
                SELECT DISTINCT field, category
                FROM argus_dismissed_findings
                WHERE report_id::text = ANY(%s)
                """,
                (run_ids,),
            )
            dismissed_rows = cur.fetchall()

        runs_considered = len(runs)
        by_key: dict[tuple[str, str], list[Any]] = {}
        for row in finding_rows:
            key = (str(row["field"]), str(row["category"]))
            by_key.setdefault(key, []).append(row)

        histories = [
            FindingHistory(
                field=k[0],
                category=k[1],
                runs_considered=runs_considered,
                runs_seen=len(rows),
                severity_history=[str(r["severity"]) for r in rows],
                affected_pct_history=[float(r["affected_pct"]) for r in rows],
                last_seen_run_at=rows[0]["run_at"],
            )
            for k, rows in by_key.items()
        ]
        histories.sort(key=lambda h: (-h.runs_seen, -h.last_seen_run_at.timestamp()))

        dismissed = [(str(r["field"]), str(r["category"])) for r in dismissed_rows]

        return HistoricalContext(
            runs_considered=runs_considered,
            last_run_at=last_run_at,
            finding_histories=histories,
            dismissed_pairs=dismissed,
        )

    def get_previous(
        self,
        *,
        collection: str,
        database: str,
        before: datetime | None = None,
    ) -> AuditReport | None:
        """Most recent report for (collection, database), strictly before ``before`` if given."""
        sql = """
            SELECT id FROM argus_reports
            WHERE collection = %s AND database = %s
        """
        args: list[Any] = [collection, database]
        if before is not None:
            sql += " AND run_at < %s"
            args.append(before)
        sql += " ORDER BY run_at DESC LIMIT 1"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, args)
            row: Any = cur.fetchone()
        if row is None:
            return None
        return self.get(UUID(str(row[0])))

    # ---- internals ---------------------------------------------------------

    def _connect(self) -> PGConnection:
        return psycopg2.connect(self._dsn)


# ----------------------------------------------------------------------------
# Row helpers — kept module-level so they're testable without a live DB.
# ----------------------------------------------------------------------------

def _insert_finding(
    cur: Any,
    report_id: str,
    f: Finding,
    *,
    dismissed: bool,
    dismiss_meta: dict[str, Any] | None,
) -> None:
    if dismissed:
        meta = dismiss_meta or {}
        cur.execute(
            """
            INSERT INTO argus_dismissed_findings (
                id, report_id, field, category, severity, description,
                dismissed_by, dismiss_verdict, dismiss_reason, dismiss_score, critique
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(f.id), report_id, f.field, f.category, f.severity.value, f.description,
                meta.get("dismissed_by", "unknown"),
                meta.get("dismiss_verdict", "fail"),
                meta.get("dismiss_reason", ""),
                float(meta.get("dismiss_score", 0.0)),
                meta.get("critique"),
            ),
        )
        return
    cur.execute(
        """
        INSERT INTO argus_findings (
            id, report_id, field, category, severity, description, hypothesis,
            evidence_query, affected_count, affected_pct, sample_values, confirmed
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            str(f.id), report_id, f.field, f.category, f.severity.value, f.description, f.hypothesis,
            f.evidence_query, f.affected_count, f.affected_pct,
            Json(_jsonable_list(f.sample_values)), f.confirmed,
        ),
    )


def _insert_eval_record(cur: Any, report_id: str, rec: EvaluationRecord) -> None:
    cur.execute(
        """
        INSERT INTO argus_evaluation_records (
            id, report_id, gate, evaluated_by, verdict, score, reason, critique, target_id, iteration
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            str(rec.id), report_id, rec.gate, rec.evaluated_by, rec.verdict.value,
            rec.score, rec.reason, rec.critique,
            str(rec.target_id) if rec.target_id else None,
            rec.iteration,
        ),
    )


def _dismiss_meta_for(report: AuditReport, finding: Finding) -> dict[str, Any]:
    """Recover the evaluator decision that dismissed this finding from the run trail."""
    for rec in reversed(report.evaluation_records):
        if rec.gate == "finding" and rec.target_id == finding.id:
            return {
                "dismissed_by": rec.evaluated_by,
                "dismiss_verdict": rec.verdict.value,
                "dismiss_reason": rec.reason,
                "dismiss_score": rec.score,
                "critique": rec.critique,
            }
    return {}


def _jsonable_list(values: list[Any]) -> list[Any]:
    """Coerce non-JSON-serializable elements (datetime, UUID) into strings."""
    out: list[Any] = []
    for v in values:
        try:
            json.dumps(v)
            out.append(v)
        except TypeError:
            out.append(str(v))
    return out
