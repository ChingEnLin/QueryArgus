"""End-to-end CLI tests with mocked Azure auth + scripted LLM + mongomock connection."""

from __future__ import annotations

import json
import sys
import types
from typing import TYPE_CHECKING, Any

import mongomock
import pytest
import typer
from typer.testing import CliRunner

from queryargus.cli.main import app
from queryargus.llm.client import ScriptedLLMClient
from queryargus.models.action import AgentAction
from queryargus.models.config import ArgusConfig
from queryargus.models.connection import CosmosConnection

if TYPE_CHECKING:
    MongoClientT = mongomock.MongoClient[dict[str, Any]]
else:
    MongoClientT = mongomock.MongoClient


@pytest.fixture
def patched_credentials(monkeypatch: pytest.MonkeyPatch) -> MongoClientT:
    """Patch ``CosmosConnection.from_default_credential`` AND the LLM factory.

    The LLM stub returns: schema_sample → conclude — minimum viable run.
    """
    client: MongoClientT = mongomock.MongoClient()
    client["audit"]["users"].insert_many(
        [
            {"_id": i, "name": f"u{i}", "age": 25 + (i % 30)} for i in range(20)
        ]
    )

    def _fake(
        cls: type[CosmosConnection],
        /,
        *,
        account: str,
        database: str,
        prefer_read_only: bool = True,
        **_: Any,
    ) -> CosmosConnection:
        return CosmosConnection.from_existing_client(client=client, cosmos_account=account, database=database)

    monkeypatch.setattr(CosmosConnection, "from_default_credential", classmethod(_fake))

    def _fake_llm(_: ArgusConfig) -> ScriptedLLMClient:
        return ScriptedLLMClient(
            actions=[
                AgentAction(reasoning="survey", action="schema_sample", action_input={"sample_size": 20}, confidence=0.95),
                AgentAction(reasoning="check ages", action="get_stats", action_input={"field": "age", "operation": "min"}, confidence=0.7),
                AgentAction(reasoning="check ages 2", action="get_stats", action_input={"field": "age", "operation": "max"}, confidence=0.7),
                AgentAction(reasoning="done", action="conclude", action_input={}, confidence=1.0),
            ]
        )

    monkeypatch.setattr("queryargus.cli.main._build_llm_client", _fake_llm)
    return client


def test_run_text_output(patched_credentials: MongoClientT) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", "--account", "my-acct", "--database", "audit", "--collection", "users", "--sample-size", "20"],
    )
    assert result.exit_code == 0, result.stdout
    assert "collection: users" in result.stdout
    # Default scripted LLM has no write_finding actions, so we expect an empty findings line.
    assert "No findings committed." in result.stdout or "FINDINGS" in result.stdout


def test_run_json_output(patched_credentials: MongoClientT) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--account", "my-acct",
            "--database", "audit",
            "--collection", "users",
            "--sample-size", "20",
            "--output", "json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    # New shape: AuditReport, not SchemaSampleResult.
    assert payload["collection"] == "users"
    assert payload["database"] == "audit"
    assert "findings" in payload
    assert "run_trace" in payload
    assert "evaluation_records" in payload


def test_inspect_text_output(patched_credentials: MongoClientT) -> None:
    """`inspect` is the bare schema-sample path — no LLM, no agent loop."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["inspect", "--account", "my-acct", "--database", "audit", "--collection", "users", "--sample-size", "10"],
    )
    assert result.exit_code == 0, result.stdout
    assert "collection: users" in result.stdout
    assert "name" in result.stdout


def test_run_surfaces_credential_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(cls: type[CosmosConnection], /, **_: Any) -> CosmosConnection:
        raise RuntimeError("Run `az login` and try again.")

    monkeypatch.setattr(CosmosConnection, "from_default_credential", classmethod(_explode))
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--account", "x", "--database", "d", "--collection", "c"])
    assert result.exit_code == 1
    assert "az login" in (result.stdout + (result.stderr or ""))


def test_accounts_command_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from queryargus.cli import main as cli_main

    monkeypatch.setattr(
        cli_main,
        "list_cosmos_accounts",
        lambda: [
            {"name": "alpha", "id": "/x/alpha", "subscription": "Sub-1", "location": "eastus", "kind": "MongoDB"},
            {"name": "beta", "id": "/x/beta", "subscription": "Sub-2", "location": "westus", "kind": "MongoDB"},
        ],
    )
    runner = CliRunner()
    result = runner.invoke(app, ["accounts"])
    assert result.exit_code == 0, result.stdout
    assert "alpha" in result.stdout
    assert "beta" in result.stdout


def test_accounts_command_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from queryargus.cli import main as cli_main

    monkeypatch.setattr(cli_main, "list_cosmos_accounts", lambda: [{"name": "alpha", "id": "/x/alpha"}])
    runner = CliRunner()
    result = runner.invoke(app, ["accounts", "--output", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["name"] == "alpha"


def test_auth_status_emulator(monkeypatch: pytest.MonkeyPatch) -> None:
    from queryargus.cli import main as cli_main

    monkeypatch.setattr(cli_main, "check_auth", lambda: {"ok": True, "mode": "emulator", "account": "emulator"})
    runner = CliRunner()
    result = runner.invoke(app, ["auth-status"])
    assert result.exit_code == 0
    assert "emulator" in result.stdout


def test_auth_status_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from queryargus.cli import main as cli_main

    monkeypatch.setattr(cli_main, "check_auth", lambda: {"ok": False, "reason": "Run `az login`"})
    runner = CliRunner()
    result = runner.invoke(app, ["auth-status"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# --cache-report (cache-lens integration)
# ---------------------------------------------------------------------------

def _install_fake_cache_lens(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the cache_lens module with a session that records lifecycle calls."""
    state: dict[str, Any] = {}

    class _FakeCacheLens:
        def __init__(self, client: Any, *, json_export: str | None = None, **_: Any) -> None:
            state["wrapped_target"] = client
            state["json_export"] = json_export

        def __enter__(self) -> Any:
            state["entered"] = True
            return state["wrapped_target"]

        def __exit__(self, *args: Any) -> bool:
            state["exited"] = True
            return False

    fake_mod = types.ModuleType("cache_lens")
    fake_mod.CacheLens = _FakeCacheLens  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cache_lens", fake_mod)
    return state


def test_build_cache_session_missing_dep_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    from queryargus.cli import main as cli_main

    # sys.modules[name] = None makes `import cache_lens` raise ImportError.
    monkeypatch.setitem(sys.modules, "cache_lens", None)
    with pytest.raises(typer.Exit) as exc_info:
        cli_main._build_cache_session(json_export=None)
    assert exc_info.value.exit_code == 2


def test_build_cache_session_missing_api_key_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    from queryargus.cli import main as cli_main

    _install_fake_cache_lens(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(typer.Exit) as exc_info:
        cli_main._build_cache_session(json_export=None)
    assert exc_info.value.exit_code == 1


def test_build_cache_session_constructs_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from queryargus.cli import main as cli_main

    state = _install_fake_cache_lens(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    session = cli_main._build_cache_session(json_export="cache.json")
    assert type(session).__name__ == "_FakeCacheLens"
    assert state["json_export"] == "cache.json"
    assert state["wrapped_target"] is not None  # the raw genai.Client
