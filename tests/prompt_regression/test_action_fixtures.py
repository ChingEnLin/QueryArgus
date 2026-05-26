"""Replay each (state, action) fixture through the real RulesActionEvaluator.

What this guards against:
- Edits to ``agent/evaluation/rules.py`` that silently change rule semantics.
- Edits to ``agent/state.py`` that break the fields the rules read
  (``has_run_query``, ``schema``, ``remaining_budget``).
- Edits to ``models/action.py`` that change the AgentAction wire shape so that
  recorded LLM outputs no longer parse.

A fixture is the contract; the evaluator output is the test.
"""

from __future__ import annotations

import pytest

from queryargus.agent.evaluation.rules import RulesActionEvaluator
from queryargus.models.evaluation import EvaluationVerdict

from .conftest import ActionFixture, load_action_fixtures

_VERDICT_MAP = {
    "pass": EvaluationVerdict.PASS,
    "warn": EvaluationVerdict.WARN,
    "fail": EvaluationVerdict.FAIL,
}


def _ids(fixtures: list[ActionFixture]) -> list[str]:
    return [f.name for f in fixtures]


@pytest.mark.prompt_regression
@pytest.mark.parametrize("fixture", load_action_fixtures(), ids=_ids(load_action_fixtures()))
def test_action_fixture_verdict(fixture: ActionFixture) -> None:
    evaluator = RulesActionEvaluator()
    result = evaluator.evaluate(fixture.action, fixture.state)

    expected = _VERDICT_MAP[fixture.expected_verdict]
    assert result.verdict == expected, (
        f"fixture={fixture.name} expected={expected.value} got={result.verdict.value} "
        f"reason={result.reason!r}"
    )

    if fixture.expected_rule is not None:
        assert fixture.expected_rule in result.reason, (
            f"fixture={fixture.name} expected rule={fixture.expected_rule!r} "
            f"to appear in reason={result.reason!r}"
        )
