"""End-to-end CLI tests with a mongomock-backed connection injected in."""

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
def patch_connection(monkeypatch: pytest.MonkeyPatch) -> MongoClientT:
    """Patch ``CosmosConnection.from_connection_string`` to return a mongomock-backed connection."""
    client: MongoClientT = mongomock.MongoClient()
    client["audit"]["users"].insert_many(
        [
            {"_id": 1, "name": "Alice", "age": 30},
            {"_id": 2, "name": "Bob", "age": None},
            {"_id": 3, "name": "Carol"},
        ]
    )

    def _fake(
        connection_string: str, *, cosmos_account: str, database: str, **_: Any
    ) -> CosmosConnection:
        return CosmosConnection.from_existing_client(
            client=client, cosmos_account=cosmos_account, database=database
        )

    monkeypatch.setattr(
        "queryargus.cli.main.CosmosConnection.from_connection_string", _fake
    )
    monkeypatch.setenv("COSMOS_CONNECTION_STRING", "mongodb://fake:fake@host/")
    return client


def test_run_text_output(patch_connection: MongoClientT) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["run", "--collection", "users", "--database", "audit", "--sample-size", "10"]
    )
    assert result.exit_code == 0, result.stdout
    assert "collection: users" in result.stdout
    assert "name" in result.stdout


def test_run_json_output(patch_connection: MongoClientT) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--collection",
            "users",
            "--database",
            "audit",
            "--sample-size",
            "10",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["collection"] == "users"
    paths = {f["path"] for f in payload["fields"]}
    assert {"name", "age"}.issubset(paths)


def test_run_missing_env_var_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COSMOS_CONNECTION_STRING", raising=False)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--collection", "users", "--database", "audit"])
    assert result.exit_code != 0
