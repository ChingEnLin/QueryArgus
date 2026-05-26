"""Golden-file snapshots of the rendered prompt.

Two layers of protection:

1. ``SYSTEM_PROMPT`` is snapshotted verbatim. Any edit to ``prompts.py`` —
   intentional or not — fails this test, forcing the author to refresh the
   snapshot and re-review the LLM-judged layer.

2. The user prompt (``render_user_prompt(state.summarize())``) is rendered for
   canonical states drawn from the action fixtures. This catches edits to
   ``state.summarize()`` or to the prompt template that change what the LLM
   actually sees, even when the system prompt is unchanged.

Refresh snapshots with::

    UPDATE_PROMPT_SNAPSHOTS=1 pytest -m prompt_regression

…then review the diff before committing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from queryargus.agent.prompts import SYSTEM_PROMPT, render_user_prompt

from .conftest import PROMPT_SNAPSHOT_DIR, load_action_fixtures

_UPDATE = os.environ.get("UPDATE_PROMPT_SNAPSHOTS") == "1"

# Fixture names used for user-prompt snapshots. Choosing two with distinct
# state shapes (empty vs post-sample with prior queries) catches most
# rendering changes without exploding the snapshot count.
_USER_PROMPT_FIXTURES = ("01_schema_sample_first", "03_duplicate_run_query")


def _assert_snapshot(actual: str, golden: Path) -> None:
    if _UPDATE or not golden.exists():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(actual)
        if not _UPDATE:
            pytest.skip(f"wrote initial snapshot to {golden} — re-run to assert")
        return
    expected = golden.read_text()
    assert actual == expected, (
        f"snapshot mismatch for {golden.name}. "
        f"Re-run with UPDATE_PROMPT_SNAPSHOTS=1 once you've reviewed the diff."
    )


@pytest.mark.prompt_regression
def test_system_prompt_snapshot() -> None:
    _assert_snapshot(SYSTEM_PROMPT, PROMPT_SNAPSHOT_DIR / "system_prompt.txt")


@pytest.mark.prompt_regression
@pytest.mark.parametrize("fixture_name", _USER_PROMPT_FIXTURES)
def test_user_prompt_snapshot(fixture_name: str) -> None:
    fixtures = {f.name: f for f in load_action_fixtures()}
    if fixture_name not in fixtures:
        pytest.fail(f"snapshot references missing fixture: {fixture_name}")
    state = fixtures[fixture_name].state
    rendered = render_user_prompt(state.summarize())
    _assert_snapshot(rendered, PROMPT_SNAPSHOT_DIR / f"user_prompt_{fixture_name}.txt")
