"""Shared pytest fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mongomock
import pytest

from queryargus.models.connection import CosmosConnection

if TYPE_CHECKING:
    MongoClientT = mongomock.MongoClient[dict[str, Any]]
else:
    MongoClientT = mongomock.MongoClient


@pytest.fixture
def mongo_client() -> MongoClientT:
    client: MongoClientT = mongomock.MongoClient()
    return client


@pytest.fixture
def connection(mongo_client: MongoClientT) -> CosmosConnection:
    return CosmosConnection.from_existing_client(
        client=mongo_client,
        cosmos_account="test-account",
        database="testdb",
    )


@pytest.fixture
def seeded_users(connection: CosmosConnection) -> CosmosConnection:
    """A 'users' collection deliberately seeded with mixed-quality documents."""
    docs: list[dict[str, Any]] = [
        {"_id": 1, "name": "Alice", "age": 30, "email": "alice@example.com",
         "profile": {"city": "NYC", "score": 0.91}},
        {"_id": 2, "name": "Bob", "age": 25, "email": "bob@example.com",
         "profile": {"city": "LA", "score": 0.55}},
        {"_id": 3, "name": "Carol", "age": None, "email": "carol@example.com",
         "profile": {"city": "NYC", "score": 0.78}},  # null age
        {"_id": 4, "name": "Dave", "age": 9999, "email": "dave@example.com",
         "profile": {"city": "Boston"}},  # outlier age, missing score
        {"_id": 5, "name": "Eve", "email": "eve@example.com",
         "profile": {"city": "Chicago", "score": 0.42}},  # missing age
        {"_id": 6, "name": "Frank", "age": "thirty", "email": "frank@example.com",
         "profile": {"city": "NYC", "score": 0.65}},  # type mismatch
        {"_id": 7, "name": "Grace", "age": 28, "tags": ["admin", "beta"]},  # missing email & profile
    ]
    connection.collection("users").insert_many(docs)
    return connection
