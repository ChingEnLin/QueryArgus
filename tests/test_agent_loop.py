"""End-to-end loop tests with a scripted LLM and mongomock-backed seeded data.

Exercises the weekend-2 done-when criteria:
1. Agent finds a deliberately-introduced data quality issue.
2. Rules evaluator blocks at least one bad agent action (repeated query / empty
   evidence) during a run.
"""

from __future__ import annotations

from typing import Any

import mongomock
import pytest

from queryargus.agent.loop import ArgusAgent
from queryargus.llm.client import ScriptedLLMClient
from queryargus.models.action import AgentAction
from queryargus.models.config import ArgusConfig
from queryargus.models.connection import CosmosConnection
from queryargus.models.evaluation import EvaluationVerdict
from queryargus.models.finding import FindingSeverity


@pytest.fixture
def dirty_connection() -> CosmosConnection:
    """Seed a 'users' collection with a deliberate null-rate issue on `email`."""
    client: mongomock.MongoClient[dict[str, Any]] = mongomock.MongoClient()
    docs: list[dict[str, Any]] = []
    for i in range(100):
        # 30% of docs have null email — well above the 5% warning threshold.
        email: str | None = None if i % 10 < 3 else f"user{i}@example.com"
        docs.append({"_id": i, "name": f"u{i}", "age": 25 + (i % 50), "email": email})
    client["audit"]["users"].insert_many(docs)
    return CosmosConnection.from_existing_client(client, cosmos_account="acct", database="audit")


def test_agent_finds_seeded_null_rate_issue(dirty_connection: CosmosConnection) -> None:
    """Done-when criterion #1: agent commits a null_rate finding on `email`."""
    actions = [
        AgentAction(reasoning="survey schema", action="schema_sample", action_input={"sample_size": 200}, confidence=0.95),
        AgentAction(
            reasoning="email null rate looks high; confirm",
            action="run_query",
            action_input={"filter": {"email": None}},
            confidence=0.9,
        ),
        AgentAction(
            reasoning="commit the null_rate finding",
            action="write_finding",
            action_input={
                "field": "email",
                "category": "null_rate",
                "severity": "high",
                "description": "Email is null on a meaningful fraction of users.",
                "hypothesis": "Optional-by-design or a broken signup writer dropping email values.",
                "evidence_query": '{"email": null}',
                "affected_count": 30,
                "affected_pct": 0.30,
                "sample_values": [None],
            },
            confidence=0.92,
        ),
        AgentAction(reasoning="other fields look ok", action="get_stats", action_input={"field": "age", "operation": "min"}, confidence=0.7),
        AgentAction(reasoning="and max", action="get_stats", action_input={"field": "age", "operation": "max"}, confidence=0.7),
        AgentAction(reasoning="done", action="conclude", action_input={}, confidence=1.0),
    ]
    llm = ScriptedLLMClient(actions)
    agent = ArgusAgent.with_defaults(config=ArgusConfig(sample_size=200, max_iterations=10), llm=llm)
    report = agent.run(connection=dirty_connection, collection="users")

    null_findings = [f for f in report.findings if f.field == "email" and f.category == "null_rate"]
    assert null_findings, f"agent did not commit a null_rate finding: {report.findings}"
    assert null_findings[0].severity == FindingSeverity.HIGH
    assert null_findings[0].affected_count == 30
    # Run gate should pass cleanly.
    assert report.run_evaluation is not None
    assert report.run_evaluation.verdict != EvaluationVerdict.FAIL


def test_action_evaluator_blocks_repeated_query(dirty_connection: CosmosConnection) -> None:
    """Done-when criterion #2: a duplicate run_query is rejected by the action gate."""
    dup_filter = {"email": None}
    actions = [
        AgentAction(reasoning="survey", action="schema_sample", action_input={"sample_size": 50}, confidence=0.95),
        AgentAction(reasoning="first query", action="run_query", action_input={"filter": dup_filter}, confidence=0.85),
        # Duplicate — should be blocked.
        AgentAction(reasoning="oops, again", action="run_query", action_input={"filter": dup_filter}, confidence=0.85),
        # New, valid query so the loop has something to do after the block.
        AgentAction(reasoning="different filter", action="run_query", action_input={"filter": {"age": {"$lt": 0}}}, confidence=0.6),
        AgentAction(reasoning="done", action="conclude", action_input={}, confidence=1.0),
    ]
    agent = ArgusAgent.with_defaults(
        config=ArgusConfig(sample_size=50, max_iterations=10),
        llm=ScriptedLLMClient(actions),
    )
    report = agent.run(connection=dirty_connection, collection="users")

    # The action gate should have logged a FAIL on the repeat.
    fails = [
        rec for rec in report.evaluation_records
        if rec.gate == "action" and rec.verdict == EvaluationVerdict.FAIL
    ]
    assert fails, f"no action FAIL recorded, evaluation_records={report.evaluation_records}"
    assert any("no_repeat_query" in rec.reason for rec in fails)
    # The duplicate filter must only appear once in queries_run (the second was blocked).
    assert sum(1 for q in report.run_trace if q.action == "run_query" and q.action_input.get("filter") == dup_filter) == 2  # both proposed
    # But mongo only saw it once — second was gated. Check via the matched count via the report.
    # (We can't introspect connection here, so we assert via evaluation_records above.)


def test_agent_dismisses_zero_evidence_finding(dirty_connection: CosmosConnection) -> None:
    """Finding gate: a write_finding with affected_count=0 is rejected and dismissed."""
    actions = [
        AgentAction(reasoning="survey", action="schema_sample", action_input={"sample_size": 50}, confidence=0.95),
        AgentAction(
            reasoning="propose a baseless finding",
            action="write_finding",
            action_input={
                "field": "name",
                "category": "imaginary",
                "severity": "low",
                "description": "I think there is something wrong here.",
                "hypothesis": "Vibes",
                "evidence_query": '{"name": "definitely-not-here"}',
                "affected_count": 0,
                "affected_pct": 0.0,
            },
            confidence=0.4,
        ),
        AgentAction(reasoning="done", action="conclude", action_input={}, confidence=1.0),
    ]
    agent = ArgusAgent.with_defaults(
        config=ArgusConfig(sample_size=50, max_iterations=10),
        llm=ScriptedLLMClient(actions),
    )
    report = agent.run(connection=dirty_connection, collection="users")

    # The finding must NOT be in committed findings — it was rejected.
    assert not any(f.category == "imaginary" for f in report.findings)
    # And it must show up in dismissed_findings (default policy is log_only).
    assert any(f.category == "imaginary" for f in report.dismissed_findings)


def test_agent_terminates_on_budget_exhaustion(dirty_connection: CosmosConnection) -> None:
    """If the LLM never says conclude, the loop still terminates at iteration_budget."""
    # Scripted LLM returns schema_sample once, then nothing — fallback action keeps the queue
    # synthetically conclude-ing via fallback. Override fallback to a non-conclude to force budget exit.
    forever = AgentAction(
        reasoning="loop forever",
        action="run_query",
        action_input={"filter": {"name": "u0"}},
        confidence=0.5,
    )
    actions = [
        AgentAction(reasoning="survey", action="schema_sample", action_input={"sample_size": 30}, confidence=0.9),
    ]
    # Each iteration after the first proposes the same filter — gate FAILs after the first run_query.
    llm = ScriptedLLMClient(actions, fallback=forever)
    agent = ArgusAgent.with_defaults(
        config=ArgusConfig(sample_size=30, max_iterations=4),
        llm=llm,
    )
    report = agent.run(connection=dirty_connection, collection="users")

    # Loop ran the full budget without hanging.
    assert len(report.run_trace) == 4
    # And a run gate verdict was produced.
    assert report.run_evaluation is not None
