"""CosmosConnection — connection abstraction the agent operates against.

Three factories:
- ``from_connection_string`` — explicit connection string (CI, emulator, local)
- ``from_env`` — read ``COSMOS_CONNECTION_STRING`` from the environment
- ``from_existing_client`` — wrap a ``MongoClient`` already authenticated by a
  host (e.g. QueryPal's OBO flow)

The DefaultAzureCredential / ARM-lookup factory is deferred — see PLAN.md §4
(weekend 2). For emulator/local the connection-string path is sufficient.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

# Mirror QueryMCPal's connection-string redaction so we never leak secrets in logs.
_CONN_STR_CRED_RE = re.compile(r"(://)([^:@/]+):([^@]+)@")
_CONN_STR_QUERY_SECRET_RE = re.compile(r"([?&])(password|accountKey|accountkey|key)=([^&]+)", re.IGNORECASE)

_DEFAULT_SERVER_SELECTION_TIMEOUT_MS = 8000


def redact_connection_string(conn_str: str) -> str:
    """Return ``conn_str`` with userinfo and known-secret query params replaced by ``***``."""
    redacted = _CONN_STR_CRED_RE.sub(r"\1\2:***@", conn_str)
    redacted = _CONN_STR_QUERY_SECRET_RE.sub(r"\1\2=***", redacted)
    return redacted


@dataclass(frozen=True)
class CosmosConnection:
    """A live, authenticated handle to a specific Cosmos DB (MongoDB API) database.

    The agent never holds a raw ``MongoClient`` — it goes through this object so
    that hosts (QueryPal) can inject their own authenticated client and the
    standalone CLI can build one from env / a connection string.
    """

    client: MongoClient[dict[str, Any]]
    cosmos_account: str
    database_name: str

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        cosmos_account: str,
        database: str,
        *,
        server_selection_timeout_ms: int = _DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
    ) -> CosmosConnection:
        """Build a connection from an explicit connection string and ping for fast-fail."""
        client: MongoClient[dict[str, Any]] = MongoClient(
            connection_string,
            serverSelectionTimeoutMS=server_selection_timeout_ms,
        )
        client.admin.command("ping")
        return cls(client=client, cosmos_account=cosmos_account, database_name=database)

    @classmethod
    def from_env(
        cls,
        cosmos_account: str,
        database: str,
        *,
        env_var: str = "COSMOS_CONNECTION_STRING",
        server_selection_timeout_ms: int = _DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
    ) -> CosmosConnection:
        """Build a connection from ``$COSMOS_CONNECTION_STRING`` (or a custom env var)."""
        conn_str = os.environ.get(env_var)
        if not conn_str:
            raise RuntimeError(
                f"{env_var} is not set; cannot build CosmosConnection from environment."
            )
        return cls.from_connection_string(
            conn_str,
            cosmos_account=cosmos_account,
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

    @property
    def database(self) -> Database[dict[str, Any]]:
        return self.client[self.database_name]

    def collection(self, name: str) -> Collection[dict[str, Any]]:
        return self.database[name]
