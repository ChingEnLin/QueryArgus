"""Azure auth + ARM-based Cosmos DB connection-string discovery.

Mirrors QueryMCPal's ``azure_service.py`` so the suite shares one idiom for
Cosmos auth. The flow:

1. The user runs ``az login`` once on their machine. ``DefaultAzureCredential``
   picks up that token cache.
2. ``_arm_token`` first honours an ``AZURE_ACCESS_TOKEN`` env override (used in
   Docker/CI where ``az`` is unavailable) and otherwise asks the credential for
   a token scoped to ARM.
3. ARM REST calls (raw ``requests``, no SDK import beyond ``azure-identity``)
   list subscriptions, list Cosmos accounts, and POST ``listConnectionStrings``
   for the chosen account. Results are cached in TTL caches to avoid hammering
   ARM on every tool call.
4. Connection strings are filtered: read-only (Mongo) preferred over read-write,
   primary preferred over secondary. The caller can override.

Set ``QUERYARGUS_EMULATOR=true`` to skip ARM entirely and use a canned local
connection string. Note: the Cosmos emulator does not support the MongoDB API
on Apple Silicon (see PLAN.md), so this flag is mostly useful on x86 Linux CI.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from functools import lru_cache
from typing import Any

import requests
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import CredentialUnavailableError, DefaultAzureCredential
from cachetools import TTLCache, cached

logger = logging.getLogger(__name__)


_EMULATOR = os.getenv("QUERYARGUS_EMULATOR", "").lower() == "true"
_EMULATOR_CONN_STR = (
    "mongodb://localhost:C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4"
    "b8mGGyPMbIZnqyMcsG9CXoGXkOKDiNZ/YSj5rtXM96whEh27Y3BKg=@localhost:10255/admin?ssl=true"
)

_ARM_SCOPE = "https://management.azure.com/.default"
_ARM_TIMEOUT_S = 15

_subscriptions_cache: TTLCache[Any, Any] = TTLCache(maxsize=1, ttl=3600)
_accounts_cache: TTLCache[Any, Any] = TTLCache(maxsize=10, ttl=3600)
_connstr_cache: TTLCache[Any, Any] = TTLCache(maxsize=20, ttl=3600)


@lru_cache(maxsize=1)
def _credential() -> DefaultAzureCredential:
    return DefaultAzureCredential()


def _arm_token() -> str:
    """Acquire a bearer token for ARM. Honours ``AZURE_ACCESS_TOKEN`` env override."""
    env_token = os.environ.get("AZURE_ACCESS_TOKEN")
    if env_token:
        return env_token
    try:
        return _credential().get_token(_ARM_SCOPE).token
    except ClientAuthenticationError as exc:
        raise RuntimeError(
            "Azure credential not found or expired. Run `az login` and try again."
        ) from exc


def _arm_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_arm_token()}"}


def _upn_from_token(token_str: str) -> str:
    """Best-effort UPN extraction from a JWT (no signature verification)."""
    try:
        payload_b64 = token_str.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload: dict[str, Any] = json.loads(base64.b64decode(payload_b64))
        upn = payload.get("upn") or payload.get("unique_name") or payload.get("email") or ""
        return str(upn) if upn else "authenticated"
    except Exception:  # noqa: BLE001 — fallback is intentional
        return "authenticated"


def check_auth() -> dict[str, Any]:
    """Side-effect-free auth probe. Safe to call any time."""
    if _EMULATOR:
        return {"ok": True, "account": "emulator", "mode": "emulator"}
    try:
        cred = DefaultAzureCredential()
        token = cred.get_token(_ARM_SCOPE)
        return {"ok": True, "account": _upn_from_token(token.token)}
    except (ClientAuthenticationError, CredentialUnavailableError):
        return {
            "ok": False,
            "reason": "Run `az login` in a terminal and try again.",
        }


@cached(_subscriptions_cache)
def list_subscriptions() -> list[dict[str, Any]]:
    url = "https://management.azure.com/subscriptions?api-version=2020-01-01"
    resp = requests.get(url, headers=_arm_headers(), timeout=_ARM_TIMEOUT_S)
    resp.raise_for_status()
    return list(resp.json().get("value", []))


@cached(_accounts_cache)
def list_cosmos_accounts() -> list[dict[str, Any]]:
    """Return all Cosmos DB accounts visible to the current credential."""
    if _EMULATOR:
        return [
            {
                "name": "local-emulator",
                "id": "emulator",
                "subscription": "local",
                "location": "localhost",
                "kind": "GlobalDocumentDB",
            }
        ]

    accounts: list[dict[str, Any]] = []
    for sub in list_subscriptions():
        sub_id = sub["subscriptionId"]
        sub_name = sub.get("displayName", sub_id)
        url = (
            f"https://management.azure.com/subscriptions/{sub_id}/resources"
            "?api-version=2021-04-01"
            "&$filter=resourceType eq 'Microsoft.DocumentDB/databaseAccounts'"
        )
        resp = requests.get(url, headers=_arm_headers(), timeout=_ARM_TIMEOUT_S)
        if resp.status_code != 200:
            logger.warning("Could not list resources in subscription %s: %s", sub_id, resp.text)
            continue
        for acct in resp.json().get("value", []):
            accounts.append(
                {
                    "name": acct["name"],
                    "id": acct["id"],
                    "subscription": sub_name,
                    "location": acct.get("location", "unknown"),
                    "kind": acct.get("kind", "GlobalDocumentDB"),
                }
            )
    return accounts


def resolve_account_id(account: str) -> str:
    """Accept either an account name or a full ARM resource ID; return the ARM ID."""
    if account.startswith("/subscriptions/"):
        return account
    matches = [a for a in list_cosmos_accounts() if a["name"] == account]
    if not matches:
        names = ", ".join(sorted(a["name"] for a in list_cosmos_accounts())) or "(none visible)"
        raise RuntimeError(
            f"Cosmos account {account!r} not found in any visible subscription. "
            f"Visible accounts: {names}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Cosmos account name {account!r} is ambiguous across subscriptions; "
            f"pass the full ARM resource ID instead."
        )
    return str(matches[0]["id"])


@cached(_connstr_cache)
def get_connection_string(account_id: str, *, prefer_read_only: bool = True) -> str:
    """Retrieve a Cosmos connection string via ARM ``listConnectionStrings``.

    By default returns the **read-only** Mongo primary string when ARM offers one,
    falling back to the read-write primary with a WARN log. This belt-and-braces
    matches the user's safety stance: even though every tool in QueryArgus is
    read-only, the credential we ride should be too.
    """
    if _EMULATOR:
        return _EMULATOR_CONN_STR

    url = (
        f"https://management.azure.com/{account_id}"
        "/listConnectionStrings?api-version=2023-03-15"
    )
    resp = requests.post(url, headers=_arm_headers(), timeout=_ARM_TIMEOUT_S)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to retrieve connection string for {account_id}: "
            f"{resp.status_code} – {resp.text}"
        )
    conn_strings: list[dict[str, Any]] = resp.json().get("connectionStrings", [])
    if not conn_strings:
        raise RuntimeError(f"No connection strings returned for account {account_id}")

    return _select_connection_string(conn_strings, prefer_read_only=prefer_read_only)


def _select_connection_string(
    conn_strings: list[dict[str, Any]], *, prefer_read_only: bool
) -> str:
    """Pick the best Mongo connection string from ARM's list."""
    def desc(cs: dict[str, Any]) -> str:
        return str(cs.get("description", "")).lower()

    mongo = [cs for cs in conn_strings if "mongo" in desc(cs)]
    pool = mongo or conn_strings  # if ARM doesn't tag mongo, take whatever is there

    if prefer_read_only:
        ro_primary = [cs for cs in pool if "primary" in desc(cs) and "read-only" in desc(cs)]
        if ro_primary:
            return str(ro_primary[0]["connectionString"])
        ro_any = [cs for cs in pool if "read-only" in desc(cs)]
        if ro_any:
            return str(ro_any[0]["connectionString"])
        logger.warning(
            "No read-only connection string returned by ARM; falling back to read-write primary. "
            "Consider rotating to a read-only key for QueryArgus dev runs."
        )

    rw_primary = [cs for cs in pool if "primary" in desc(cs) and "read-only" not in desc(cs)]
    if rw_primary:
        return str(rw_primary[0]["connectionString"])
    return str(pool[0]["connectionString"])
