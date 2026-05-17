"""Tests for cross-run memory: HistoricalContext + ReportStore.load_history + state.summarize integration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import mongomock
import pytest

from queryargus.agent.loop import ArgusAgent
from queryargus.agent.state import AgentState
from queryargus.llm.client import ScriptedLLMClient
from queryargus.models.action import AgentAction
from queryargus.models.config import ArgusConfig
from queryargus.models.connection import CosmosConnection
from queryargus.models.history import (
    DismissedPattern,
    FindingHistory,
    HistoricalContext,
    UserVerdictHistory,
    empty_history,
)
from queryargus.storage import ReportStore

# ---------------------------------------------------------------------------
# FindingHistory + HistoricalContext pure-data behaviour
# ---------------------------------------------------------------------------

def test_finding_history_persistent_when_seen_twice() -> None:
    h = FindingHistory(
        field="email", category="null_rate", runs_considered=5, runs_seen=2,
        severity_history=["high", "high"], affected_pct_history=[0.30, 0.32],
        last_seen_run_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    assert h.is_persistent
    assert h.is_stable
    assert h.last_severity == "high"


def test_finding_history_drifting_when_pct_swings() -> None:
    h = FindingHistory(
        field="email", category="null_rate", runs_considered=3, runs_seen=3,
        severity_history=["high", "medium", "high"],
        affected_pct_history=[0.50, 0.10, 0.30],   # max-min = 0.40 > stability range
        last_seen_run_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    assert h.is_persistent
    assert not h.is_stable


def test_empty_history_renders_to_empty_string() -> None:
    ctx = empty_history()
    assert ctx.is_empty
    assert ctx.render() == ""


def test_render_includes_persistent_one_off_and_dismissed() -> None:
    persistent = FindingHistory(
        field="email", category="null_rate", runs_considered=4, runs_seen=4,
        severity_history=["high"] * 4, affected_pct_history=[0.30, 0.31, 0.32, 0.30],
        last_seen_run_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    one_off = FindingHistory(
        field="age", category="outlier_value", runs_considered=4, runs_seen=1,
        severity_history=["medium"], affected_pct_history=[0.02],
        last_seen_run_at=datetime(2026, 4, 1, tzinfo=UTC),
    )
    ctx = HistoricalContext(
        runs_considered=4,
        last_run_at=datetime(2026, 5, 1, tzinfo=UTC),
        finding_histories=[persistent, one_off],
        dismissed_patterns=[
            DismissedPattern(
                field="imaging_data",
                category="type_mismatch",
                dismiss_reason="evidence_query did not match the described condition",
                critique='use {"$or": [{"$exists": false}, ...]}',
            ),
        ],
    )
    rendered = ctx.render()
    assert "PERSISTENT FINDINGS" in rendered
    assert "email / null_rate" in rendered
    assert "stable" in rendered
    assert "ONE-OFF FINDINGS" in rendered
    assert "age / outlier_value" in rendered
    assert "DISMISSED PATTERNS" in rendered
    assert "imaging_data / type_mismatch" in rendered
    # The critique is what closes the cross-run learning loop — it MUST be in the prompt.
    assert "rejected because" in rendered
    assert "evidence_query did not match" in rendered
    assert "suggested correction" in rendered


def test_render_marks_persistent_plus_dismissed_for_re_proposal() -> None:
    """When a (field, category) is BOTH persistent and dismissed, the prompt must escalate."""
    persistent = FindingHistory(
        field="versioning.base_patient_id", category="null_rate",
        runs_considered=3, runs_seen=3,
        severity_history=["high"] * 3, affected_pct_history=[0.49, 0.78, 0.98],
        last_seen_run_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    ctx = HistoricalContext(
        runs_considered=3,
        last_run_at=datetime(2026, 5, 1, tzinfo=UTC),
        finding_histories=[persistent],
        dismissed_patterns=[
            DismissedPattern(
                field="versioning.base_patient_id", category="null_rate",
                dismiss_reason="query checked null only but description claimed null-or-missing",
                critique='use $or with $exists: false',
            ),
        ],
    )
    rendered = ctx.render()
    assert "NOTE" in rendered
    assert "also listed as PERSISTENT" in rendered
    assert "qualitatively stronger evidence" in rendered


# ---------------------------------------------------------------------------
# AgentState.summarize integration
# ---------------------------------------------------------------------------

def test_state_summarize_omits_history_block_when_none() -> None:
    state = AgentState(collection="c", database="d", cosmos_account="a", iteration_budget=20)
    assert "HISTORICAL CONTEXT" not in state.summarize()


def test_state_summarize_includes_history_block_when_present() -> None:
    h = FindingHistory(
        field="email", category="null_rate", runs_considered=3, runs_seen=3,
        severity_history=["high", "high", "high"],
        affected_pct_history=[0.30, 0.31, 0.32],
        last_seen_run_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    state = AgentState(
        collection="c", database="d", cosmos_account="a", iteration_budget=20,
        historical_context=HistoricalContext(
            runs_considered=3,
            last_run_at=datetime(2026, 5, 1, tzinfo=UTC),
            finding_histories=[h],
        ),
    )
    summary = state.summarize()
    assert "HISTORICAL CONTEXT" in summary
    assert "email / null_rate" in summary


# ---------------------------------------------------------------------------
# ReportStore.load_history with a fake cursor
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, scripted: list[list[Any]]) -> None:
        self.scripted = list(scripted)  # one batch per execute call
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, sql: str, args: Any = ()) -> None:
        self.executed.append((sql, tuple(args) if isinstance(args, list | tuple) else (args,)))

    def fetchall(self) -> list[Any]:
        return self.scripted.pop(0) if self.scripted else []

    def fetchone(self) -> Any | None:
        batch = self.scripted.pop(0) if self.scripted else []
        return batch[0] if batch else None

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self, cursor_factory: Any = None) -> _FakeCursor:
        return self._cursor

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def test_load_history_aggregates_findings_across_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    rid_a, rid_b, rid_c = uuid4(), uuid4(), uuid4()
    now = datetime(2026, 5, 1, tzinfo=UTC)
    cursor = _FakeCursor(
        scripted=[
            # 1) last-N runs query
            [
                {"id": rid_a, "run_at": now},
                {"id": rid_b, "run_at": now - timedelta(days=1)},
                {"id": rid_c, "run_at": now - timedelta(days=2)},
            ],
            # 2) findings query — UNION of committed and dismissed.
            # email committed in all 3 runs; age committed in 1; vfield dismissed in 2.
            [
                {"field": "email", "category": "null_rate", "severity": "high", "affected_pct": 0.30,
                 "run_id": rid_a, "run_at": now, "source": "committed"},
                {"field": "email", "category": "null_rate", "severity": "high", "affected_pct": 0.31,
                 "run_id": rid_b, "run_at": now - timedelta(days=1), "source": "committed"},
                {"field": "email", "category": "null_rate", "severity": "high", "affected_pct": 0.32,
                 "run_id": rid_c, "run_at": now - timedelta(days=2), "source": "committed"},
                {"field": "age", "category": "outlier_value", "severity": "medium", "affected_pct": 0.02,
                 "run_id": rid_c, "run_at": now - timedelta(days=2), "source": "committed"},
                # vfield was dismissed in 2 runs — it should still rank as persistent.
                {"field": "vfield", "category": "null_rate", "severity": "high", "affected_pct": None,
                 "run_id": rid_a, "run_at": now, "source": "dismissed"},
                {"field": "vfield", "category": "null_rate", "severity": "high", "affected_pct": None,
                 "run_id": rid_b, "run_at": now - timedelta(days=1), "source": "dismissed"},
            ],
            # 3) dismissed query (DISTINCT ON returns latest per (field, category) with critique)
            [
                {
                    "field": "imaging_data",
                    "category": "type_mismatch",
                    "dismiss_reason": "evidence_query did not match the description",
                    "critique": "use $or with $exists: false",
                    "created_at": now,
                }
            ],
            # 4) user-verdict query — no labelled findings in this scenario
            [],
        ]
    )
    monkeypatch.setattr(
        "queryargus.storage.postgres.psycopg2.connect",
        lambda *_a, **_kw: _FakeConn(cursor),
    )

    store = ReportStore(dsn="postgresql://fake/fake")
    ctx = store.load_history(collection="users", database="db", limit=3)

    assert ctx.runs_considered == 3
    assert ctx.last_run_at == now

    # Three distinct (field, category) keys: email committed 3x, age committed 1x,
    # vfield dismissed 2x. vfield must rank as PERSISTENT despite never committing —
    # this is what closes the cross-run learning loop.
    assert len(ctx.finding_histories) == 3
    persistent = ctx.persistent_findings
    one_off = ctx.one_off_findings
    persistent_keys = {(p.field, p.category) for p in persistent}
    assert ("email", "null_rate") in persistent_keys
    assert ("vfield", "null_rate") in persistent_keys     # ← the load-bearing assertion
    assert len(one_off) == 1
    assert one_off[0].field == "age"

    # vfield has no affected_pct entries (only dismissed appearances) → not stable.
    vfield_history = next(h for h in persistent if h.field == "vfield")
    assert vfield_history.runs_seen == 2
    assert not vfield_history.is_stable
    assert vfield_history.affected_pct_history == []

    assert len(ctx.dismissed_patterns) == 1
    pattern = ctx.dismissed_patterns[0]
    assert pattern.field == "imaging_data"
    assert pattern.category == "type_mismatch"
    assert "did not match" in pattern.dismiss_reason
    assert pattern.critique == "use $or with $exists: false"


def test_load_history_empty_when_no_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor(scripted=[[]])  # last-N runs query returns nothing
    monkeypatch.setattr(
        "queryargus.storage.postgres.psycopg2.connect",
        lambda *_a, **_kw: _FakeConn(cursor),
    )
    store = ReportStore(dsn="postgresql://fake/fake")
    ctx = store.load_history(collection="x", database="y")
    assert ctx.is_empty


# ---------------------------------------------------------------------------
# User verdicts (Arm A — post-hoc rating)
# ---------------------------------------------------------------------------

def test_user_verdict_net_label() -> None:
    tp_dominant = UserVerdictHistory(
        field="email", category="null_rate",
        tp_count=3, fp_count=1, last_label="tp",
        last_labelled_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    assert tp_dominant.net_label == "tp"

    fp_dominant = UserVerdictHistory(
        field="cohort", category="enum_violation",
        tp_count=0, fp_count=2, last_label="fp",
        last_labelled_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    assert fp_dominant.net_label == "fp"

    mixed = UserVerdictHistory(
        field="other", category="type_drift",
        tp_count=1, fp_count=1, last_label="fp",
        last_labelled_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    assert mixed.net_label is None


def test_render_annotates_persistent_with_user_verdict() -> None:
    """A persistent FP-net (field, category) must be tagged inline in the prompt."""
    persistent = FindingHistory(
        field="cohort", category="enum_violation",
        runs_considered=4, runs_seen=4,
        severity_history=["medium"] * 4,
        affected_pct_history=[0.05, 0.05, 0.05, 0.05],
        last_seen_run_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    ctx = HistoricalContext(
        runs_considered=4,
        last_run_at=datetime(2026, 5, 1, tzinfo=UTC),
        finding_histories=[persistent],
        user_verdicts=[
            UserVerdictHistory(
                field="cohort", category="enum_violation",
                tp_count=0, fp_count=3, last_label="fp",
                last_labelled_at=datetime(2026, 5, 1, tzinfo=UTC),
            )
        ],
    )
    rendered = ctx.render()
    # Inline suffix on the persistent line — the planner reads top-down, so this is
    # the load-bearing place to surface the FP signal.
    assert "USER-MARKED FP" in rendered
    assert "require stronger evidence" in rendered


def test_render_includes_orphan_user_verdicts_block() -> None:
    """Verdicts on (field, category) NOT in this run's findings/dismissed must still surface."""
    ctx = HistoricalContext(
        runs_considered=2,
        last_run_at=datetime(2026, 5, 1, tzinfo=UTC),
        finding_histories=[],
        dismissed_patterns=[],
        user_verdicts=[
            UserVerdictHistory(
                field="archived_at", category="stale_timestamp",
                tp_count=2, fp_count=0, last_label="tp",
                last_labelled_at=datetime(2026, 5, 1, tzinfo=UTC),
            )
        ],
    )
    rendered = ctx.render()
    assert "USER VERDICTS on prior findings not surfaced" in rendered
    assert "archived_at / stale_timestamp" in rendered
    assert "net=tp" in rendered


def test_load_history_includes_user_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:
    rid = uuid4()
    now = datetime(2026, 5, 1, tzinfo=UTC)
    cursor = _FakeCursor(
        scripted=[
            # 1) last-N runs
            [{"id": rid, "run_at": now}],
            # 2) findings UNION dismissed — empty so we can isolate verdict path
            [],
            # 3) latest-dismissal-per-key — empty
            [],
            # 4) user-verdict query
            [
                {
                    "field": "cohort",
                    "category": "enum_violation",
                    "tp_count": 1,
                    "fp_count": 3,
                    "last_label": "fp",
                    "last_labelled_at": now,
                },
                {
                    "field": "archived_at",
                    "category": "stale_timestamp",
                    "tp_count": 2,
                    "fp_count": 0,
                    "last_label": "tp",
                    "last_labelled_at": now,
                },
            ],
        ]
    )
    monkeypatch.setattr(
        "queryargus.storage.postgres.psycopg2.connect",
        lambda *_a, **_kw: _FakeConn(cursor),
    )
    store = ReportStore(dsn="postgresql://fake/fake")
    ctx = store.load_history(collection="users", database="db")

    assert len(ctx.user_verdicts) == 2
    by_field = {v.field: v for v in ctx.user_verdicts}
    assert by_field["cohort"].net_label == "fp"
    assert by_field["cohort"].fp_count == 3
    assert by_field["archived_at"].net_label == "tp"
    assert by_field["archived_at"].tp_count == 2


def test_update_user_label_writes_relational_and_raw_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sets argus_findings.user_label AND patches the matching entry in raw_report JSONB."""
    report_id, finding_id = uuid4(), uuid4()

    class _UpdateCursor:
        def __init__(self) -> None:
            self.executed: list[tuple[str, tuple[Any, ...]]] = []
            self.rowcount = 0
            self._select_raw = {
                "findings": [
                    {"id": str(finding_id), "user_label": None, "field": "f"},
                    {"id": str(uuid4()), "user_label": None, "field": "other"},
                ]
            }
            self._next_one: Any = None

        def execute(self, sql: str, args: Any = ()) -> None:
            self.executed.append((sql, tuple(args) if isinstance(args, list | tuple) else (args,)))
            if sql.strip().startswith("UPDATE argus_findings"):
                self.rowcount = 1
            elif sql.strip().startswith("SELECT raw_report"):
                # psycopg2 returns a tuple here (we don't use RealDictCursor on this path)
                self._next_one = (self._select_raw,)
            elif sql.strip().startswith("UPDATE argus_reports"):
                # capture so the assertion below can inspect the patched JSONB
                pass

        def fetchone(self) -> Any | None:
            value, self._next_one = self._next_one, None
            return value

        def __enter__(self) -> _UpdateCursor:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    cursor = _UpdateCursor()
    monkeypatch.setattr(
        "queryargus.storage.postgres.psycopg2.connect",
        lambda *_a, **_kw: _FakeConn(cursor),  # type: ignore[arg-type]
    )

    store = ReportStore(dsn="postgresql://fake/fake")
    ok = store.update_user_label(report_id=report_id, finding_id=finding_id, label="fp")
    assert ok is True
    # raw_report mutation applied in place before the UPDATE.
    target = next(f for f in cursor._select_raw["findings"] if f["id"] == str(finding_id))
    assert target["user_label"] == "fp"
    # Other findings untouched.
    other = next(f for f in cursor._select_raw["findings"] if f["id"] != str(finding_id))
    assert other["user_label"] is None
    sqls = [sql.strip().split()[0:3] for sql, _ in cursor.executed]
    # Sequence: UPDATE argus_findings → SELECT raw_report → UPDATE argus_reports
    assert sqls[0][:2] == ["UPDATE", "argus_findings"]
    assert sqls[1][:2] == ["SELECT", "raw_report"]
    assert sqls[2][:2] == ["UPDATE", "argus_reports"]


def test_update_user_label_returns_false_when_finding_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MissCursor:
        def __init__(self) -> None:
            self.executed: list[Any] = []
            self.rowcount = 0  # UPDATE matches nothing

        def execute(self, sql: str, args: Any = ()) -> None:
            self.executed.append((sql, args))

        def fetchone(self) -> Any:
            return None

        def __enter__(self) -> _MissCursor:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    cursor = _MissCursor()
    monkeypatch.setattr(
        "queryargus.storage.postgres.psycopg2.connect",
        lambda *_a, **_kw: _FakeConn(cursor),  # type: ignore[arg-type]
    )
    store = ReportStore(dsn="postgresql://fake/fake")
    assert store.update_user_label(report_id=uuid4(), finding_id=uuid4(), label="tp") is False
    # Only the relational UPDATE should fire; the raw_report path is short-circuited.
    assert len(cursor.executed) == 1


# ---------------------------------------------------------------------------
# End-to-end: history flows from store → agent → prompt the LLM sees
# ---------------------------------------------------------------------------

def test_agent_loop_sees_history_in_prompt() -> None:
    """Prove the agent's prompt actually includes HISTORICAL CONTEXT when one is passed."""
    client: mongomock.MongoClient[dict[str, Any]] = mongomock.MongoClient()
    client["db"]["c"].insert_many([{"_id": i, "x": i} for i in range(3)])
    conn = CosmosConnection.from_existing_client(client, cosmos_account="acct", database="db")

    h = FindingHistory(
        field="x", category="null_rate", runs_considered=2, runs_seen=2,
        severity_history=["high", "high"], affected_pct_history=[0.3, 0.31],
        last_seen_run_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    history = HistoricalContext(
        runs_considered=2,
        last_run_at=datetime(2026, 5, 1, tzinfo=UTC),
        finding_histories=[h],
    )

    llm = ScriptedLLMClient(
        actions=[
            AgentAction(reasoning="survey", action="schema_sample", action_input={"sample_size": 3}, confidence=0.95),
            AgentAction(reasoning="done", action="conclude", action_input={}, confidence=1.0),
        ]
    )
    agent = ArgusAgent.with_defaults(config=ArgusConfig(sample_size=3, max_iterations=2), llm=llm)
    agent.run(connection=conn, collection="c", history=history)

    # Both prompts should contain the historical-context block.
    assert llm.prompts, "scripted LLM should have been called"
    for _system, user in llm.prompts:
        assert "HISTORICAL CONTEXT" in user
        assert "x / null_rate" in user
