"""Loaders and helpers for the prompt regression fixture suite.

Fixtures live as JSON files under ``fixtures/actions/`` and ``fixtures/findings/``.
This module turns a fixture dict into the same object graph the live agent
holds at evaluation time: a real ``AgentState`` (with an optional
``SchemaSampleResult`` and prior ``queries_run``), a real ``AgentAction`` /
``Finding``. The tests then exercise the production ``RulesActionEvaluator`` /
``RulesFindingEvaluator`` against those objects.

Keeping the loader here (not in test files) means new fixtures only need the
JSON file — no Python changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from queryargus.agent.state import AgentState
from queryargus.models.action import AgentAction
from queryargus.models.finding import Finding
from queryargus.tools.schema_sample import FieldStats, SchemaSampleResult

FIXTURES_ROOT = Path(__file__).parent / "fixtures"
ACTION_FIXTURES_DIR = FIXTURES_ROOT / "actions"
FINDING_FIXTURES_DIR = FIXTURES_ROOT / "findings"
PROMPT_SNAPSHOT_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class ActionFixture:
    name: str
    description: str
    state: AgentState
    action: AgentAction
    expected_verdict: str
    expected_rule: str | None


@dataclass(frozen=True)
class FindingFixture:
    name: str
    description: str
    state: AgentState
    finding: Finding
    expected_verdict: str
    expected_rule: str | None


def _build_state(raw: dict[str, Any]) -> AgentState:
    """Build an AgentState from a permissive dict.

    Only the fields used by the deterministic rules and by ``state.summarize()``
    are honoured. Anything not provided falls back to the AgentState defaults
    (empty schema, no queries run, no history).
    """
    schema_raw = raw.get("schema")
    schema: SchemaSampleResult | None
    if schema_raw is None:
        schema = None
    else:
        schema = SchemaSampleResult(
            collection=schema_raw.get("collection", raw.get("collection", "test")),
            documents_sampled=schema_raw.get("documents_sampled", 0),
            fields=[FieldStats.model_validate(f) for f in schema_raw.get("fields", [])],
            truncated_paths=schema_raw.get("truncated_paths", []),
        )

    return AgentState(
        collection=raw.get("collection", "test_coll"),
        database=raw.get("database", "test_db"),
        cosmos_account=raw.get("cosmos_account", "test-account"),
        iteration_budget=raw.get("iteration_budget", 10),
        iteration=raw.get("iteration", 0),
        documents_sampled=raw.get("documents_sampled", 0),
        collection_size=raw.get("collection_size", 0),
        schema=schema,
        queries_run=list(raw.get("queries_run", [])),
        fields_investigated=set(raw.get("fields_investigated", [])),
        fields_concluded=set(raw.get("fields_concluded", [])),
    )


def _load_dir(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.json"))
    out: list[dict[str, Any]] = []
    for path in files:
        with path.open() as fh:
            data = json.load(fh)
        data.setdefault("name", path.stem)
        out.append(data)
    return out


def load_action_fixtures() -> list[ActionFixture]:
    out: list[ActionFixture] = []
    for data in _load_dir(ACTION_FIXTURES_DIR):
        out.append(
            ActionFixture(
                name=data["name"],
                description=data.get("description", ""),
                state=_build_state(data.get("state", {})),
                action=AgentAction.model_validate(data["action"]),
                expected_verdict=data["expected_verdict"],
                expected_rule=data.get("expected_rule"),
            )
        )
    return out


def load_finding_fixtures() -> list[FindingFixture]:
    out: list[FindingFixture] = []
    for data in _load_dir(FINDING_FIXTURES_DIR):
        out.append(
            FindingFixture(
                name=data["name"],
                description=data.get("description", ""),
                state=_build_state(data.get("state", {})),
                finding=Finding.model_validate(data["finding"]),
                expected_verdict=data["expected_verdict"],
                expected_rule=data.get("expected_rule"),
            )
        )
    return out
