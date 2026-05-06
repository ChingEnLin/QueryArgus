"""get_stats — aggregate stats for a single field via the MongoDB aggregation pipeline."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from queryargus.models.connection import CosmosConnection

logger = logging.getLogger(__name__)

StatsOperation = Literal["count", "min", "max", "avg", "distinct"]

_DISTINCT_HARD_CAP = 1000


class StatsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: str
    field: str
    operation: StatsOperation
    result: Any
    query_used: str = Field(description="Human-readable description of the aggregation that ran.")


def get_stats(
    connection: CosmosConnection,
    collection_name: str,
    field: str,
    operation: StatsOperation,
    *,
    filter: dict[str, Any] | None = None,  # noqa: A002 — mongo terminology
) -> StatsResult:
    """Compute one of count/min/max/avg/distinct for ``field``.

    ``distinct`` is hard-capped at ``_DISTINCT_HARD_CAP`` distinct values to bound
    response size; if more values exist, the agent should switch to a different
    investigation strategy (e.g. ``run_query`` with a filter).
    """
    if not field:
        raise ValueError("field must be a non-empty path")

    coll = connection.collection(collection_name)
    match_stage: list[dict[str, Any]] = [{"$match": filter}] if filter else []
    field_ref = f"${field}"
    query_used: str
    result: Any

    if operation == "count":
        pipeline = [
            *match_stage,
            {"$match": {field: {"$exists": True}}},
            {"$count": "n"},
        ]
        rows = list(coll.aggregate(pipeline))
        result = rows[0]["n"] if rows else 0
        query_used = f"count of docs where '{field}' exists"

    elif operation in {"min", "max", "avg"}:
        op_key = f"${operation}"
        pipeline = [
            *match_stage,
            {"$match": {field: {"$exists": True, "$ne": None}}},
            {"$group": {"_id": None, "v": {op_key: field_ref}}},
        ]
        rows = list(coll.aggregate(pipeline))
        result = rows[0]["v"] if rows else None
        query_used = f"{operation} of '{field}' over docs where it is non-null"

    elif operation == "distinct":
        pipeline = [
            *match_stage,
            {"$match": {field: {"$exists": True}}},
            {"$group": {"_id": field_ref}},
            {"$limit": _DISTINCT_HARD_CAP},
        ]
        rows = list(coll.aggregate(pipeline))
        result = [row["_id"] for row in rows]
        query_used = f"distinct values of '{field}' (capped at {_DISTINCT_HARD_CAP})"

    else:  # pragma: no cover — Literal exhausts at type level
        raise ValueError(f"unknown operation: {operation}")

    logger.info(
        "get_stats collection=%s field=%s op=%s",
        collection_name,
        field,
        operation,
    )
    return StatsResult(
        collection=collection_name,
        field=field,
        operation=operation,
        result=result,
        query_used=query_used,
    )
