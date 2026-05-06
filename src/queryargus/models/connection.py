"""CosmosConnection — connection abstraction the agent operates against.

Three factories:

- ``from_default_credential`` — primary path. Uses ``DefaultAzureCredential``
  (i.e. ``az login`` token cache) to look up the account's connection string
  via ARM, then opens a ``MongoClient``. Mirrors QueryMCPal.
- ``from_existing_client`` — wrap a ``MongoClient`` already authenticated by a
  host (QueryPal's OBO flow injects this).
- ``from_connection_string`` — emulator / CI escape hatch. Use only when you
  can't run ``DefaultAzureCredential`` (CI without a token override, local
  emulator).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

_CONN_STR_CRED_RE = re.compile(r"(://)([^:@/]+):([^@]+)@")
_CONN_STR_QUERY_SECRET_RE = re.compile(
    r"([?&])(password|accountKey|accountkey|key)=([^&]+)", re.IGNORECASE
)

_DEFAULT_SERVER_SELECTION_TIMEOUT_MS = 8000


def redact_connection_string(conn_str: str) -> str:
    """Return ``conn_str`` with userinfo and known-secret query params replaced by ``***``."""
    redacted = _CONN_STR_CRED_RE.sub(r"\1\2:***@", conn_str)
    redacted = _CONN_STR_QUERY_SECRET_RE.sub(r"\1\2=***", redacted)
    return redacted


@dataclass(frozen=True)
class CosmosConnection:
    """A live, authenticated handle to a specific Cosmos DB (MongoDB API) database."""

    client: MongoClient[dict[str, Any]]
    cosmos_account: str
    database_name: str

    @classmethod
    def from_default_credential(
        cls,
        account: str,
        database: str,
        *,
        prefer_read_only: bool = True,
        server_selection_timeout_ms: int = _DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
    ) -> CosmosConnection:
        """Authenticate via ``az login`` / ``DefaultAzureCredential`` and fetch the conn string from ARM.

        ``account`` accepts either a plain account name (resolved via ARM) or a
        full ``/subscriptions/.../databaseAccounts/<name>`` resource ID.
        """
        # Local import: avoids a cycle and keeps the heavy azure-identity imports
        # off the critical path of from_existing_client / from_connection_string.
        from queryargus.azure_service import get_connection_string, resolve_account_id

        account_id = resolve_account_id(account)
        conn_str = get_connection_string(account_id, prefer_read_only=prefer_read_only)
        return cls.from_connection_string(
            conn_str,
            cosmos_account=account,
            database=database,
            server_selection_timeout_ms=server_selection_timeout_ms,
        )

    @classmethod
    def from_existing_client(
        cls,
        client: MongoClient[dict[str, Any]],
        cosmos_account: str,
        database: str,
    ) -> CosmosConnection:
        """Wrap a ``MongoClient`` provided by the host (e.g. QueryPal OBO flow)."""
        return cls(client=client, cosmos_account=cosmos_account, database_name=database)

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        cosmos_account: str,
        database: str,
        *,
        server_selection_timeout_ms: int = _DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
    ) -> CosmosConnection:
        """Open a ``MongoClient`` from an explicit connection string and ping for fast-fail."""
        client: MongoClient[dict[str, Any]] = MongoClient(
            connection_string,
            serverSelectionTimeoutMS=server_selection_timeout_ms,
        )
        client.admin.command("ping")
        return cls(client=client, cosmos_account=cosmos_account, database_name=database)

    @property
    def database(self) -> Database[dict[str, Any]]:
        return self.client[self.database_name]

    def collection(self, name: str) -> Collection[dict[str, Any]]:
        return self.database[name]
