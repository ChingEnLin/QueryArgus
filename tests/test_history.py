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
from queryargus.models.history import FindingHistory, HistoricalContext, empty_history
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
        dismissed_pairs=[("imaging_data", "type_mismatch")],
    )
    rendered = ctx.render()
    assert "PERSISTENT FINDINGS" in rendered
    assert "email / null_rate" in rendered
    assert "stable" in rendered
    assert "ONE-OFF FINDINGS" in rendered
    assert "age / outlier_value" in rendered
    assert "DISMISSED PATTERNS" in rendered
    assert "imaging_data / type_mismatch" in rendered


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
            # 2) findings query — email seen in all 3, age seen in 1
            [
                {"field": "email", "category": "null_rate", "severity": "high", "affected_pct": 0.30, "run_at": now},
                {"field": "email", "category": "null_rate", "severity": "high", "affected_pct": 0.31, "run_at": now - timedelta(days=1)},
                {"field": "email", "category": "null_rate", "severity": "high", "affected_pct": 0.32, "run_at": now - timedelta(days=2)},
                {"field": "age", "category": "outlier_value", "severity": "medium", "affected_pct": 0.02, "run_at": now - timedelta(days=2)},
            ],
            # 3) dismissed query
            [{"field": "imaging_data", "category": "type_mismatch"}],
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

    # Two distinct (field, category) keys; email is persistent (seen 3x), age is one-off.
    assert len(ctx.finding_histories) == 2
    persistent = ctx.persistent_findings
    one_off = ctx.one_off_findings
    assert len(persistent) == 1
    assert persistent[0].field == "email"
    assert persistent[0].runs_seen == 3
    assert persistent[0].is_stable
    assert len(one_off) == 1
    assert one_off[0].field == "age"

    assert ctx.dismissed_pairs == [("imaging_data", "type_mismatch")]


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
