"""Tests for the schema_sample tool."""

from __future__ import annotations

from queryargus.models.connection import CosmosConnection
from queryargus.tools.schema_sample import _MAX_DEPTH, schema_sample


def _stats_by_path(result, path: str):  # type: ignore[no-untyped-def]
    for f in result.fields:
        if f.path == path:
            return f
    raise AssertionError(f"path {path!r} not found in {[f.path for f in result.fields]}")


def test_samples_documents_and_returns_field_stats(seeded_users: CosmosConnection) -> None:
    result = schema_sample(seeded_users, "users", sample_size=100)

    assert result.collection == "users"
    assert result.documents_sampled == 7

    name = _stats_by_path(result, "name")
    assert name.present_count == 7
    assert name.missing_count == 0
    assert name.types == {"str": 7}
    assert name.cardinality == 7  # all distinct

    age = _stats_by_path(result, "age")
    # Carol has null, Eve has missing (not in doc)
    assert age.present_count == 6, age
    assert age.null_count == 1
    assert age.missing_count == 1
    # Type spread: int + null + str ('thirty')
    assert "int" in age.types
    assert "null" in age.types
    assert "str" in age.types


def test_flattens_nested_objects_with_dot_notation(seeded_users: CosmosConnection) -> None:
    result = schema_sample(seeded_users, "users", sample_size=100)

    city = _stats_by_path(result, "profile.city")
    assert city.present_count == 6  # Grace has no profile
    assert city.missing_count == 1
    assert city.types == {"str": 6}

    score = _stats_by_path(result, "profile.score")
    # Dave has no score, Grace has no profile → 2 missing
    assert score.present_count == 5
    assert score.missing_count == 2


def test_arrays_record_element_type_without_descending(seeded_users: CosmosConnection) -> None:
    result = schema_sample(seeded_users, "users", sample_size=100)
    tags = _stats_by_path(result, "tags")
    assert tags.present_count == 1  # only Grace
    # Element-type signature, not interior-flattening
    assert any(t.startswith("array<") for t in tags.types)


def test_missing_field_count_corrected_for_late_appearing_paths(connection: CosmosConnection) -> None:
    """Path 'late' first appears in doc 3 — docs 1 & 2 must be counted as missing."""
    coll = connection.collection("c")
    coll.insert_many(
        [
            {"_id": 1, "a": 1},
            {"_id": 2, "a": 2},
            {"_id": 3, "a": 3, "late": "yes"},
        ]
    )
    result = schema_sample(connection, "c", sample_size=10)
    late = _stats_by_path(result, "late")
    assert late.present_count == 1
    assert late.missing_count == 2


def test_recursion_depth_guard(connection: CosmosConnection) -> None:
    """A pathologically deep document is truncated, not crashed."""
    deep: dict[str, object] = {"v": 1}
    for _ in range(_MAX_DEPTH + 5):
        deep = {"n": deep}
    connection.collection("deep").insert_one({"_id": 1, "root": deep})

    result = schema_sample(connection, "deep", sample_size=10)
    # The walk should have stopped before unbounded depth — at least one truncated path recorded.
    assert result.truncated_paths
    # Sanity: every recorded path is shallower than the absolute pathological depth.
    for f in result.fields:
        assert f.path.count(".") <= _MAX_DEPTH


def test_sample_size_must_be_positive(connection: CosmosConnection) -> None:
    import pytest

    connection.collection("c").insert_one({"_id": 1, "a": 1})
    with pytest.raises(ValueError, match="positive"):
        schema_sample(connection, "c", sample_size=0)
