"""Tests for CosmosConnection (factories + redaction)."""

from __future__ import annotations

from typing import Any

import mongomock
import pytest

from queryargus.models.connection import CosmosConnection, redact_connection_string


def test_from_existing_client_wraps_client() -> None:
    client: mongomock.MongoClient[dict[str, Any]] = mongomock.MongoClient()
    conn = CosmosConnection.from_existing_client(client, cosmos_account="acct", database="db1")
    assert conn.cosmos_account == "acct"
    assert conn.database_name == "db1"
    assert conn.collection("users") is client["db1"]["users"]


def test_from_default_credential_resolves_via_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    """``from_default_credential`` should resolve the account, fetch the conn string,
    and hand it to ``from_connection_string`` (which we stub out so no real Mongo I/O happens)."""
    from queryargus import azure_service

    monkeypatch.setattr(azure_service, "resolve_account_id", lambda name: f"/arm/{name}")

    captured: dict[str, Any] = {}

    def fake_get_connection_string(account_id: str, *, prefer_read_only: bool) -> str:
        captured["account_id"] = account_id
        captured["prefer_read_only"] = prefer_read_only
        return "mongodb://fake-conn"

    monkeypatch.setattr(azure_service, "get_connection_string", fake_get_connection_string)

    fake_client: mongomock.MongoClient[dict[str, Any]] = mongomock.MongoClient()

    def fake_from_conn_str(connection_string: str, *, cosmos_account: str, database: str, **_: Any) -> CosmosConnection:
        captured["connection_string"] = connection_string
        captured["cosmos_account"] = cosmos_account
        captured["database"] = database
        return CosmosConnection.from_existing_client(fake_client, cosmos_account=cosmos_account, database=database)

    monkeypatch.setattr(
        CosmosConnection,
        "from_connection_string",
        classmethod(lambda cls, *a, **k: fake_from_conn_str(*a, **k)),
    )

    conn = CosmosConnection.from_default_credential(account="my-acct", database="db1")
    assert conn.cosmos_account == "my-acct"
    assert captured["account_id"] == "/arm/my-acct"
    assert captured["prefer_read_only"] is True
    assert captured["connection_string"] == "mongodb://fake-conn"


def test_from_default_credential_passes_through_read_write_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from queryargus import azure_service

    monkeypatch.setattr(azure_service, "resolve_account_id", lambda name: f"/arm/{name}")
    captured: dict[str, Any] = {}

    def fake_get_connection_string(account_id: str, *, prefer_read_only: bool) -> str:
        captured["prefer_read_only"] = prefer_read_only
        return "mongodb://fake"

    monkeypatch.setattr(azure_service, "get_connection_string", fake_get_connection_string)
    monkeypatch.setattr(
        CosmosConnection,
        "from_connection_string",
        classmethod(
            lambda cls, *a, **k: CosmosConnection.from_existing_client(
                mongomock.MongoClient(),
                cosmos_account=k.get("cosmos_account") or a[1],
                database=k.get("database") or a[2],
            )
        ),
    )

    CosmosConnection.from_default_credential(account="x", database="db1", prefer_read_only=False)
    assert captured["prefer_read_only"] is False


def test_redact_userinfo() -> None:
    s = "mongodb://user:supersecret@host:27017/db?ssl=true"
    redacted = redact_connection_string(s)
    assert "supersecret" not in redacted
    assert "user:***" in redacted


def test_redact_account_key_query_param() -> None:
    s = "mongodb://host:10255/?ssl=true&accountKey=abcd1234EFGH=="
    redacted = redact_connection_string(s)
    assert "abcd1234" not in redacted
    assert "accountKey=***" in redacted


def test_redact_no_secret_is_pass_through() -> None:
    s = "mongodb://host:27017/db"
    assert redact_connection_string(s) == s


def test_no_from_env_factory() -> None:
    """``from_env`` was removed — credential flow replaces conn-string-from-env."""
    assert not hasattr(CosmosConnection, "from_env")
