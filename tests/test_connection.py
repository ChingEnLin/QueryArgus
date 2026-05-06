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


def test_from_env_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COSMOS_CONNECTION_STRING", raising=False)
    with pytest.raises(RuntimeError, match="COSMOS_CONNECTION_STRING"):
        CosmosConnection.from_env(cosmos_account="acct", database="db1")


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
