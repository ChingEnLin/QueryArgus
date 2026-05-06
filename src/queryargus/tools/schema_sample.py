"""schema_sample — survey a collection's implicit schema.

Samples N documents via ``$sample`` and walks each one to record per-field stats:
type distribution, null/missing counts, cardinality (capped), and up to a few
sample values.

**v1 limitations** (deliberate, see PLAN.md and conversation 2026-05-05):

- Nested dicts ARE flattened with dot-notation (``user.profile.age``).
- Arrays of scalars: recorded as ``array`` with the element-type breakdown.
- Arrays of objects: NOT descended into. The element type is recorded as
  ``array<object>`` and a sample of element shapes is kept as a value sample;
  the agent can investigate array contents via targeted ``run_query`` calls.
- Recursion is bounded at ``_MAX_DEPTH`` to defend against pathological cycles.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from queryargus.models.connection import CosmosConnection

logger = logging.getLogger(__name__)

_MAX_DEPTH = 20
_CARDINALITY_CAP = 100
_SAMPLE_VALUES_PER_FIELD = 5


class FieldStats(BaseModel):
    """Aggregated statistics for one flattened field path."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    types: dict[str, int] = Field(default_factory=dict)
    present_count: int = Field(default=0, ge=0)
    null_count: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)
    cardinality: int = Field(default=0, ge=0)
    cardinality_capped: bool = False
    sample_values: list[Any] = Field(default_factory=list)

    @property
    def total_observed(self) -> int:
        return self.present_count + self.missing_count

    @property
    def null_rate(self) -> float:
        total = self.total_observed
        return 0.0 if total == 0 else self.null_count / total


class SchemaSampleResult(BaseModel):
    """Output of one ``schema_sample`` call."""

    model_config = ConfigDict(extra="forbid")

    collection: str
    documents_sampled: int = Field(ge=0)
    fields: list[FieldStats]
    truncated_paths: list[str] = Field(
        default_factory=list,
        description="Paths whose recursion hit _MAX_DEPTH and were not descended further.",
    )


def _type_name(value: Any) -> str:
    """Return a stable type name for a sampled value."""
    if value is None:
        return "null"
    if isinstance(value, bool):  # check before int — bool is an int subclass
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _array_element_type(arr: list[Any]) -> str:
    """Describe an array's element-type signature without descending into objects."""
    if not arr:
        return "array<empty>"
    elem_types = sorted({_type_name(el) for el in arr})
    if len(elem_types) == 1:
        return f"array<{elem_types[0]}>"
    return f"array<{'|'.join(elem_types)}>"


def _is_hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


class _StatsAccumulator:
    """Internal helper: builds FieldStats from a stream of (path, value) observations.

    Tracks per-path distinct values up to ``_CARDINALITY_CAP`` and sample values up
    to ``_SAMPLE_VALUES_PER_FIELD``. Unhashable values (lists, dicts) skip the
    distinct-set but still count toward presence/types and may be sampled.
    """

    def __init__(self) -> None:
        self._stats: dict[str, FieldStats] = {}
        self._distinct: dict[str, set[Any]] = {}

    def record_present(self, path: str, value: Any) -> None:
        stats = self._stats.setdefault(path, FieldStats(path=path))
        stats.present_count += 1
        type_name = _type_name(value)

        if isinstance(value, list):
            type_name = _array_element_type(value)

        stats.types[type_name] = stats.types.get(type_name, 0) + 1

        if value is None:
            stats.null_count += 1
            return

        if _is_hashable(value):
            distinct = self._distinct.setdefault(path, set())
            if len(distinct) < _CARDINALITY_CAP:
                distinct.add(value)
            else:
                stats.cardinality_capped = True

        try:
            already_sampled = value in stats.sample_values
        except TypeError:  # unhashable / unequal-comparable types
            already_sampled = False
        if len(stats.sample_values) < _SAMPLE_VALUES_PER_FIELD and not already_sampled:
            stats.sample_values.append(value)

    def finalize(self, total_docs: int) -> list[FieldStats]:
        for path, distinct in self._distinct.items():
            self._stats[path].cardinality = len(distinct)
        # Missing = docs that did not contain this path. Computed after the full
        # walk so paths first seen in later docs still account for earlier docs.
        for stats in self._stats.values():
            stats.missing_count = total_docs - stats.present_count
        return sorted(self._stats.values(), key=lambda s: s.path)


def _walk_document(
    doc: dict[str, Any],
    acc: _StatsAccumulator,
    *,
    truncated: set[str],
    parent_path: str = "",
    depth: int = 0,
) -> None:
    if depth >= _MAX_DEPTH:
        if parent_path:
            truncated.add(parent_path)
        return

    for key, value in doc.items():
        path = f"{parent_path}.{key}" if parent_path else key
        if isinstance(value, dict):
            acc.record_present(path, value)
            _walk_document(value, acc, truncated=truncated, parent_path=path, depth=depth + 1)
        else:
            acc.record_present(path, value)


def schema_sample(
    connection: CosmosConnection,
    collection_name: str,
    *,
    sample_size: int = 200,
) -> SchemaSampleResult:
    """Sample ``sample_size`` random documents and return per-field statistics.

    Uses MongoDB's ``$sample`` aggregation stage. Cosmos DB's MongoDB API supports
    this on supported account tiers; for unsupported configurations the caller
    should fall back to ``find().limit()`` (not implemented in v1).
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    coll = connection.collection(collection_name)
    cursor = coll.aggregate([{"$sample": {"size": sample_size}}], allowDiskUse=True)

    acc = _StatsAccumulator()
    truncated: set[str] = set()
    docs_seen = 0

    for doc in cursor:
        docs_seen += 1
        _walk_document(doc, acc, truncated=truncated)

    fields = acc.finalize(total_docs=docs_seen)
    logger.info(
        "schema_sample collection=%s docs_sampled=%d distinct_paths=%d truncated=%d",
        collection_name,
        docs_seen,
        len(fields),
        len(truncated),
    )
    return SchemaSampleResult(
        collection=collection_name,
        documents_sampled=docs_seen,
        fields=fields,
        truncated_paths=sorted(truncated),
    )
