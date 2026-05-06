"""End-to-end CLI tests with mocked Azure auth + a mongomock-backed connection."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import mongomock
import pytest
from typer.testing import CliRunner

from queryargus.cli.main import app
from queryargus.models.connection import CosmosConnection

if TYPE_CHECKING:
    MongoClientT = mongomock.MongoClient[dict[str, Any]]
else:
    MongoClientT = mongomock.MongoClient


@pytest.fixture
def patched_credentials(monkeypatch: pytest.MonkeyPatch) -> MongoClientT:
    """Patch ``CosmosConnection.from_default_credential`` to return a mongomock-backed connection."""
    client: MongoClientT = mongomock.MongoClient()
    client["audit"]["users"].insert_many(
        [
            {"_id": 1, "name": "Alice", "age": 30},
            {"_id": 2, "name": "Bob", "age": None},
            {"_id": 3, "name": "Carol"},
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
    return client


def test_run_text_output(patched_credentials: MongoClientT) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", "--account", "my-acct", "--database", "audit", "--collection", "users", "--sample-size", "10"],
    )
    assert result.exit_code == 0, result.stdout
    assert "collection: users" in result.stdout
    assert "name" in result.stdout


def test_run_json_output(patched_credentials: MongoClientT) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--account", "my-acct",
            "--database", "audit",
            "--collection", "users",
            "--sample-size", "10",
            "--output", "json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["collection"] == "users"
    paths = {f["path"] for f in payload["fields"]}
    assert {"name", "age"}.issubset(paths)


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
