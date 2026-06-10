"""GeminiClient construction: injected client vs self-built client."""

from __future__ import annotations

from typing import Any

import pytest

from queryargus.llm.gemini import GeminiClient


class _FakeResponse:
    text = '{"reasoning": "x", "action": "conclude", "action_input": {}, "confidence": 1.0}'
    usage_metadata = None


class _FakeModels:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: str, config: Any) -> _FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._response


class _FakeGenaiClient:
    def __init__(self) -> None:
        self.models = _FakeModels(_FakeResponse())


def test_injected_client_skips_api_key_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    fake = _FakeGenaiClient()
    gc = GeminiClient(client=fake)
    resp = gc.propose_action(system="sys", user="usr")
    assert resp.action.action == "conclude"
    assert len(fake.models.calls) == 1
    assert fake.models.calls[0]["contents"] == "usr"


def test_missing_key_and_no_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiClient()
