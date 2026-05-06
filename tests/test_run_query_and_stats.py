"""Tests for run_query and get_stats."""

from __future__ import annotations

import pytest

from queryargus.models.connection import CosmosConnection
from queryargus.tools.get_stats import get_stats
from queryargus.tools.run_query import _MAX_LIMIT, run_query


def test_run_query_returns_matched_count_and_truncation(seeded_users: CosmosConnection) -> None:
    # Five users have profile.city; ask for 2 to force truncation.
    result = run_query(
        seeded_users,
        "users",
        filter={"profile.city": {"$exists": True}},
        limit=2,
    )
    assert result.matched_count == 6
    assert result.returned_count == 2
    assert result.truncated is True


def test_run_query_no_truncation_when_under_limit(seeded_users: CosmosConnection) -> None:
    result = run_query(
        seeded_users,
        "users",
        filter={"name": "Alice"},
        limit=10,
    )
    assert result.matched_count == 1
    assert result.truncated is False


def test_run_query_caps_limit(seeded_users: CosmosConnection) -> None:
    """Asking for an absurd limit silently caps to _MAX_LIMIT — no error, just bounded reads."""
    result = run_query(seeded_users, "users", filter={}, limit=_MAX_LIMIT * 10)
    assert result.returned_count <= _MAX_LIMIT


def test_run_query_rejects_zero_limit(seeded_users: CosmosConnection) -> None:
    with pytest.raises(ValueError):
        run_query(seeded_users, "users", filter={}, limit=0)


def test_get_stats_count(seeded_users: CosmosConnection) -> None:
    result = get_stats(seeded_users, "users", "age", "count")
    # 6 docs have 'age' field present (Eve has no age key at all).
    assert result.result == 6


def test_get_stats_min_max_avg_on_clean_numeric_field(connection: CosmosConnection) -> None:
    """min/max/avg compute correctly on a clean numeric field, skipping nulls and missing."""
    # Use a fresh, type-clean collection — real Cosmos handles mixed-type ordering via BSON
    # rules, but mongomock raises TypeError on str-vs-int. We test only the typed-clean path
    # here; real-Cosmos integration tests will cover heterogeneous fields.
    coll = connection.collection("scores")
    coll.insert_many(
        [
            {"_id": 1, "score": 10},
            {"_id": 2, "score": 20},
            {"_id": 3, "score": 90},
            {"_id": 4, "score": None},  # excluded by $ne: null
            {"_id": 5},                  # excluded by $exists
        ]
    )
    assert get_stats(connection, "scores", "score", "min").result == 10
    assert get_stats(connection, "scores", "score", "max").result == 90
    assert get_stats(connection, "scores", "score", "avg").result == 40


def test_get_stats_distinct_returns_unique_values(seeded_users: CosmosConnection) -> None:
    result = get_stats(seeded_users, "users", "profile.city", "distinct")
    cities = set(result.result)
    assert {"NYC", "LA", "Boston", "Chicago"}.issubset(cities)


def test_get_stats_rejects_empty_field(seeded_users: CosmosConnection) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        get_stats(seeded_users, "users", "", "count")
