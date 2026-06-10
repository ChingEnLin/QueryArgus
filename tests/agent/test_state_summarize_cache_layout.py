"""Cache-friendliness contract for ``AgentState.summarize()``.

The prompt is structured as a *stable prefix* (fixed for the whole run once the
schema is sampled) followed by a *volatile trailer* (changes every iteration).
Keeping the byte-identical block first lets Gemini's implicit cache — and the
cachelens prefix detector — anchor on it. These tests pin that contract so a
future edit can't silently move volatile content back into the prefix.
"""

from __future__ import annotations

from queryargus.agent.state import AgentState
from queryargus.models.action import AgentAction
from queryargus.tools.schema_sample import FieldStats, SchemaSampleResult

_STABLE_HEADER = "=== COLLECTION UNDER AUDIT (fixed context) ==="
_TRAILER_HEADER = "=== INVESTIGATION PROGRESS"


def _state_with_schema(**overrides: object) -> AgentState:
    schema = SchemaSampleResult(
        collection="patient",
        documents_sampled=200,
        fields=[
            FieldStats.model_validate(
                {"path": "age", "present_count": 200, "missing_count": 0,
                 "null_count": 5, "types": {"int": 195, "null": 5}, "cardinality": 60},
            ),
            FieldStats.model_validate(
                {"path": "email", "present_count": 180, "missing_count": 20,
                 "null_count": 0, "types": {"str": 180}, "cardinality": 180},
            ),
        ],
        truncated_paths=[],
    )
    base: dict[str, object] = dict(
        collection="patient", database="dataflow", cosmos_account="acct",
        iteration_budget=20, iteration=7,
        documents_sampled=200, collection_size=50000,
        schema=schema,
    )
    base.update(overrides)
    return AgentState(**base)  # type: ignore[arg-type]


def test_stable_block_precedes_volatile_trailer() -> None:
    state = _state_with_schema()
    summary = state.summarize()
    assert _STABLE_HEADER in summary
    assert _TRAILER_HEADER in summary
    assert summary.index(_STABLE_HEADER) < summary.index(_TRAILER_HEADER)
    # Schema (stable) must sit inside the prefix, before the trailer.
    assert summary.index("SCHEMA") < summary.index(_TRAILER_HEADER)


def test_running_token_tally_is_not_in_prompt() -> None:
    # The per-call token tally changed every turn and poisoned the cache prefix.
    state = _state_with_schema()
    state.total_usage = state.total_usage  # no-op; just exercise the field
    assert "tokens used so far" not in state.summarize()


def test_iteration_counter_lives_in_the_trailer_not_the_prefix() -> None:
    state = _state_with_schema()
    summary = state.summarize()
    # The iteration number is volatile, so it must appear at/after the trailer
    # header, never inside the stable prefix.
    assert summary.index(_TRAILER_HEADER) < summary.index("7/20")


def test_schema_block_excludes_volatile_progress_markers() -> None:
    # Investigated/concluded status mutates during a run; it must not be inlined
    # into the (otherwise frozen) schema lines.
    state = _state_with_schema(
        fields_investigated={"age"}, fields_concluded={"age"},
    )
    summary = state.summarize()
    schema_section = summary[summary.index("SCHEMA"):summary.index(_TRAILER_HEADER)]
    assert "[investigated]" not in schema_section
    assert "[concluded]" not in schema_section


def test_investigated_and_concluded_fields_surface_in_trailer() -> None:
    state = _state_with_schema(
        fields_investigated={"age", "email"}, fields_concluded={"age"},
    )
    summary = state.summarize()
    trailer = summary[summary.index(_TRAILER_HEADER):]
    assert "INVESTIGATED FIELDS" in trailer
    assert "age" in trailer and "email" in trailer
    assert "CONCLUDED FIELDS" in trailer


def test_stable_prefix_is_byte_identical_as_run_progresses() -> None:
    """Two iterations of the same run must share an identical stable prefix.

    This is the core caching invariant: only the trailer may differ between
    turns once the schema is fixed.
    """
    early = _state_with_schema(iteration=3)
    early.queries_run = [{"age": {"$gt": 150}}]
    early.history = [AgentAction(reasoning="r", action="schema_sample",
                                 action_input={}, confidence=0.9)]

    late = _state_with_schema(iteration=11)
    late.queries_run = [{"age": {"$gt": 150}}, {"email": None}]
    late.history = [
        AgentAction(reasoning="r", action="schema_sample", action_input={}, confidence=0.9),
        AgentAction(reasoning="r2", action="run_query", action_input={}, confidence=0.8),
    ]
    late.fields_investigated = {"age"}

    early_prefix = early.summarize().split(_TRAILER_HEADER)[0]
    late_prefix = late.summarize().split(_TRAILER_HEADER)[0]
    assert early_prefix == late_prefix
