from __future__ import annotations

from typing import Any

import mongomock
import pytest

from queryargus.agent.loop import ArgusAgent
from queryargus.llm.client import ScriptedLLMClient, TokenUsage
from queryargus.models.action import AgentAction
from queryargus.models.config import ArgusConfig
from queryargus.models.connection import CosmosConnection
from queryargus.observability.cost import CostTracker
from tests.observability.test_observer import RecordingObserver


@pytest.fixture
def trivial_connection() -> CosmosConnection:
    client: mongomock.MongoClient[dict[str, Any]] = mongomock.MongoClient()
    client["audit"]["things"].insert_many([{"_id": i, "x": i} for i in range(5)])
    return CosmosConnection.from_existing_client(
        client, cosmos_account="acct", database="audit"
    )


def _conclude_only_llm() -> ScriptedLLMClient:
    return ScriptedLLMClient(
        actions=[
            AgentAction(
                reasoning="done", action="conclude", action_input={}, confidence=1.0
            )
        ],
        usage_per_call=TokenUsage(input_tokens=10, output_tokens=5),
        model="gemini-2.5-flash",
    )


def test_default_run_uses_null_observer(trivial_connection: CosmosConnection) -> None:
    llm = _conclude_only_llm()
    agent = ArgusAgent.with_defaults(config=ArgusConfig(max_iterations=3), llm=llm)
    report = agent.run(trivial_connection, "things")
    assert report.cost is None  # no observer attached


def test_recording_observer_sees_run_lifecycle(
    trivial_connection: CosmosConnection,
) -> None:
    llm = _conclude_only_llm()
    rec = RecordingObserver()
    agent = ArgusAgent.with_defaults(
        config=ArgusConfig(max_iterations=3), llm=llm, observers=[rec]
    )
    report = agent.run(trivial_connection, "things")

    hooks = [name for name, _ in rec.calls]
    assert hooks[0] == "on_run_start"
    assert "on_iteration_start" in hooks
    assert "on_llm_call" in hooks
    assert "on_action" in hooks
    assert hooks[-1] == "on_run_complete"

    last_name, last_kwargs = rec.calls[-1]
    assert last_name == "on_run_complete"
    assert last_kwargs["report"] is report


def test_cost_tracker_populates_report(trivial_connection: CosmosConnection) -> None:
    llm = ScriptedLLMClient(
        actions=[
            AgentAction(
                reasoning="done", action="conclude", action_input={}, confidence=1.0
            )
        ],
        usage_per_call=TokenUsage(input_tokens=1_000_000, output_tokens=0),
        model="gemini-2.5-flash",
    )
    tracker = CostTracker()
    agent = ArgusAgent.with_defaults(
        config=ArgusConfig(max_iterations=2), llm=llm, observers=[tracker]
    )
    report = agent.run(trivial_connection, "things")
    assert report.cost is not None
    assert report.cost.usd_total > 0
    assert any(m.model == "gemini-2.5-flash" for m in report.cost.by_model)


def test_tool_call_hook_emitted_for_conclude(
    trivial_connection: CosmosConnection,
) -> None:
    llm = _conclude_only_llm()
    rec = RecordingObserver()
    agent = ArgusAgent.with_defaults(
        config=ArgusConfig(max_iterations=2), llm=llm, observers=[rec]
    )
    agent.run(trivial_connection, "things")
    tool_calls = [kw for name, kw in rec.calls if name == "on_tool_call"]
    assert any(
        tc["name"] == "conclude" and tc["ok"] is True and tc["error"] is None
        for tc in tool_calls
    )
