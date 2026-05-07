"""Tests for token usage tracking through the planner -> state -> report flow."""

from __future__ import annotations

from typing import Any

import mongomock

from queryargus.agent.loop import ArgusAgent
from queryargus.llm.client import ScriptedLLMClient, TokenUsage
from queryargus.models.action import AgentAction
from queryargus.models.config import ArgusConfig
from queryargus.models.connection import CosmosConnection


def test_token_usage_addition() -> None:
    a = TokenUsage(input_tokens=10, output_tokens=2)
    b = TokenUsage(input_tokens=30, output_tokens=5)
    total = a + b
    assert total.input_tokens == 40
    assert total.output_tokens == 7
    assert total.total_tokens == 47


def test_token_usage_zero_default() -> None:
    u = TokenUsage()
    assert u.total_tokens == 0


def _connection() -> CosmosConnection:
    client: mongomock.MongoClient[dict[str, Any]] = mongomock.MongoClient()
    client["db"]["c"].insert_many([{"_id": i, "x": i} for i in range(5)])
    return CosmosConnection.from_existing_client(client, cosmos_account="acct", database="db")


def test_planner_accumulates_usage_across_iterations() -> None:
    """Each LLM call adds to AgentState.total_usage; the audit report sees the sum.

    ``max_iterations=2`` prevents the run gate's continue policy from extending
    the loop after the early conclude — we want a deterministic 2-call run.
    """
    actions = [
        AgentAction(reasoning="survey", action="schema_sample", action_input={"sample_size": 5}, confidence=0.95),
        AgentAction(reasoning="done", action="conclude", action_input={}, confidence=1.0),
    ]
    llm = ScriptedLLMClient(actions, usage_per_call=TokenUsage(input_tokens=100, output_tokens=20))
    agent = ArgusAgent.with_defaults(config=ArgusConfig(sample_size=5, max_iterations=2), llm=llm)
    report = agent.run(connection=_connection(), collection="c")

    assert report.total_input_tokens == 200
    assert report.total_output_tokens == 40


def test_zero_usage_when_client_does_not_report() -> None:
    """ScriptedLLMClient with default zero usage produces zeroed report fields."""
    actions = [
        AgentAction(reasoning="survey", action="schema_sample", action_input={"sample_size": 5}, confidence=0.95),
        AgentAction(reasoning="done", action="conclude", action_input={}, confidence=1.0),
    ]
    llm = ScriptedLLMClient(actions)
    agent = ArgusAgent.with_defaults(config=ArgusConfig(sample_size=5, max_iterations=2), llm=llm)
    report = agent.run(connection=_connection(), collection="c")
    assert report.total_input_tokens == 0
    assert report.total_output_tokens == 0
