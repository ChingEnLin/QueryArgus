"""Tests for azure_service — token resolution, ARM REST calls, conn-string preference."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from queryargus import azure_service


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """TTLCache state leaks across tests — wipe before every test."""
    azure_service._subscriptions_cache.clear()
    azure_service._accounts_cache.clear()
    azure_service._connstr_cache.clear()
    azure_service._credential.cache_clear()


def _resp(status: int, payload: Any) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.text = str(payload)
    r.raise_for_status = MagicMock()
    if status >= 400:
        r.raise_for_status.side_effect = RuntimeError(f"HTTP {status}")
    return r


def test_arm_token_prefers_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_ACCESS_TOKEN", "env-token-123")
    # Even if credential explodes, env wins.
    monkeypatch.setattr(
        azure_service, "_credential", lambda: (_ for _ in ()).throw(RuntimeError("should not be called"))
    )
    assert azure_service._arm_token() == "env-token-123"


def test_arm_token_falls_back_to_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_ACCESS_TOKEN", raising=False)
    fake_cred = MagicMock()
    fake_cred.get_token.return_value = MagicMock(token="cred-token-abc")
    monkeypatch.setattr(azure_service, "_credential", lambda: fake_cred)
    assert azure_service._arm_token() == "cred-token-abc"
    fake_cred.get_token.assert_called_once_with("https://management.azure.com/.default")


def test_list_cosmos_accounts_aggregates_across_subscriptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(azure_service, "_arm_token", lambda: "fake")
    calls: list[str] = []

    def fake_get(url: str, **_: Any) -> MagicMock:
        calls.append(url)
        if "subscriptions?" in url:
            return _resp(200, {"value": [{"subscriptionId": "S1", "displayName": "Sub One"}]})
        return _resp(
            200,
            {
                "value": [
                    {
                        "name": "acct-a",
                        "id": "/subscriptions/S1/.../databaseAccounts/acct-a",
                        "location": "eastus",
                        "kind": "MongoDB",
                    }
                ]
            },
        )

    monkeypatch.setattr("queryargus.azure_service.requests.get", fake_get)

    accts = azure_service.list_cosmos_accounts()
    assert len(accts) == 1
    assert accts[0]["name"] == "acct-a"
    assert accts[0]["subscription"] == "Sub One"
    # Two ARM calls: one for subs, one for resources within S1.
    assert len(calls) == 2


def test_resolve_account_id_passes_through_arm_id(monkeypatch: pytest.MonkeyPatch) -> None:
    full_id = "/subscriptions/S/resourceGroups/RG/providers/Microsoft.DocumentDB/databaseAccounts/x"
    # Should not need ARM at all to resolve a full ID.
    monkeypatch.setattr(
        azure_service, "list_cosmos_accounts", lambda: (_ for _ in ()).throw(AssertionError("not called"))
    )
    assert azure_service.resolve_account_id(full_id) == full_id


def test_resolve_account_id_finds_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        azure_service,
        "list_cosmos_accounts",
        lambda: [
            {"name": "alpha", "id": "/subscriptions/S/.../alpha"},
            {"name": "beta", "id": "/subscriptions/S/.../beta"},
        ],
    )
    assert azure_service.resolve_account_id("beta") == "/subscriptions/S/.../beta"


def test_resolve_account_id_raises_for_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        azure_service, "list_cosmos_accounts", lambda: [{"name": "alpha", "id": "/subscriptions/S/.../alpha"}]
    )
    with pytest.raises(RuntimeError, match="not found"):
        azure_service.resolve_account_id("missing")


def _connstr_payload(*items: tuple[str, str]) -> dict[str, Any]:
    return {"connectionStrings": [{"description": d, "connectionString": c} for d, c in items]}


def test_get_connection_string_prefers_read_only_primary_mongo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(azure_service, "_arm_token", lambda: "fake")
    payload = _connstr_payload(
        ("Primary MongoDB Connection String", "mongodb://rw-primary"),
        ("Secondary MongoDB Connection String", "mongodb://rw-secondary"),
        ("Primary Read-Only MongoDB Connection String", "mongodb://ro-primary"),
        ("Secondary Read-Only MongoDB Connection String", "mongodb://ro-secondary"),
    )
    monkeypatch.setattr("queryargus.azure_service.requests.post", lambda *a, **k: _resp(200, payload))

    got = azure_service.get_connection_string("/subscriptions/S/.../x", prefer_read_only=True)
    assert got == "mongodb://ro-primary"


def test_get_connection_string_warns_and_falls_back_when_no_read_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(azure_service, "_arm_token", lambda: "fake")
    payload = _connstr_payload(
        ("Primary MongoDB Connection String", "mongodb://rw-primary"),
        ("Secondary MongoDB Connection String", "mongodb://rw-secondary"),
    )
    monkeypatch.setattr("queryargus.azure_service.requests.post", lambda *a, **k: _resp(200, payload))

    with caplog.at_level("WARNING", logger="queryargus.azure_service"):
        got = azure_service.get_connection_string("/subscriptions/S/.../x", prefer_read_only=True)
    assert got == "mongodb://rw-primary"
    assert any("read-only" in rec.message for rec in caplog.records)


def test_get_connection_string_explicit_rw_takes_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(azure_service, "_arm_token", lambda: "fake")
    payload = _connstr_payload(
        ("Primary MongoDB Connection String", "mongodb://rw-primary"),
        ("Primary Read-Only MongoDB Connection String", "mongodb://ro-primary"),
    )
    monkeypatch.setattr("queryargus.azure_service.requests.post", lambda *a, **k: _resp(200, payload))

    got = azure_service.get_connection_string("/subscriptions/S/.../x", prefer_read_only=False)
    assert got == "mongodb://rw-primary"


def test_emulator_short_circuits_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(azure_service, "_EMULATOR", True)

    def _boom(*_: Any, **__: Any) -> None:
        raise AssertionError("ARM should not be called in emulator mode")

    monkeypatch.setattr("queryargus.azure_service.requests.post", _boom)
    got = azure_service.get_connection_string.__wrapped__(  # bypass cache
        "/whatever", prefer_read_only=True
    )
    assert got == azure_service._EMULATOR_CONN_STR


def test_check_auth_emulator_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(azure_service, "_EMULATOR", True)
    status = azure_service.check_auth()
    assert status["ok"] is True
    assert status["mode"] == "emulator"
