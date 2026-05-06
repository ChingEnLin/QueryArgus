"""run_query — bounded ``find()`` wrapper for targeted hypothesis confirmation."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from queryargus.models.connection import CosmosConnection

logger = logging.getLogger(__name__)

_DEFAULT_LIMIT = 50
_MAX_LIMIT = 1000


class RunQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: str
    filter: dict[str, Any]
    documents: list[dict[str, Any]]
    matched_count: int = Field(ge=0, description="Total docs matching the filter (pre-limit).")
    returned_count: int = Field(ge=0)
    truncated: bool = Field(description="True if matched_count > returned_count.")


def run_query(
    connection: CosmosConnection,
    collection_name: str,
    filter: dict[str, Any],  # noqa: A002 — mongo terminology, not the builtin
    *,
    projection: dict[str, Any] | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> RunQueryResult:
    """Run a bounded ``find`` and return the matching docs plus the total match count.

    ``limit`` is hard-capped at ``_MAX_LIMIT`` to keep the agent from running away
    with cursor reads. ``matched_count`` is computed independently via
    ``count_documents`` so the agent can still reason about the full match size
    even when the document list is truncated.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    effective_limit = min(limit, _MAX_LIMIT)

    coll = connection.collection(collection_name)
    matched_count = coll.count_documents(filter)

    cursor = coll.find(filter, projection=projection).limit(effective_limit)
    documents = list(cursor)

    truncated = matched_count > len(documents)
    logger.info(
        "run_query collection=%s matched=%d returned=%d truncated=%s",
        collection_name,
        matched_count,
        len(documents),
        truncated,
    )
    return RunQueryResult(
        collection=collection_name,
        filter=filter,
        documents=documents,
        matched_count=matched_count,
        returned_count=len(documents),
        truncated=truncated,
    )
