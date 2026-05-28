from __future__ import annotations

from queryargus.llm.client import JSONResponse, LLMResponse, ScriptedLLMClient, TokenUsage
from queryargus.models.action import AgentAction


def test_llm_response_carries_model_default_empty():
    resp = LLMResponse(action=AgentAction(reasoning="x", action="conclude", confidence=1.0))
    assert resp.model == ""


def test_json_response_carries_model_default_empty():
    resp = JSONResponse(raw="{}")
    assert resp.model == ""


def test_scripted_client_can_set_model_on_responses():
    client = ScriptedLLMClient(
        actions=[AgentAction(reasoning="x", action="conclude", confidence=1.0)],
        json_responses=['{"verdict": "pass", "score": 1.0, "reason": "ok", "evaluated_by": "x"}'],
        model="test-model",
        usage_per_call=TokenUsage(input_tokens=10, output_tokens=20),
    )
    a = client.propose_action(system="s", user="u")
    j = client.complete_json(system="s", user="u")
    assert a.model == "test-model"
    assert j.model == "test-model"
