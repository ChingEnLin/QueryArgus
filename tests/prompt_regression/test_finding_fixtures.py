"""Replay each (state, finding) fixture through the real RulesFindingEvaluator."""

from __future__ import annotations

import pytest

from queryargus.agent.evaluation.rules import RulesFindingEvaluator
from queryargus.models.evaluation import EvaluationVerdict

from .conftest import FindingFixture, load_finding_fixtures

_VERDICT_MAP = {
    "pass": EvaluationVerdict.PASS,
    "warn": EvaluationVerdict.WARN,
    "fail": EvaluationVerdict.FAIL,
}


def _ids(fixtures: list[FindingFixture]) -> list[str]:
    return [f.name for f in fixtures]


@pytest.mark.prompt_regression
@pytest.mark.parametrize("fixture", load_finding_fixtures(), ids=_ids(load_finding_fixtures()))
def test_finding_fixture_verdict(fixture: FindingFixture) -> None:
    evaluator = RulesFindingEvaluator()
    result = evaluator.evaluate(fixture.finding, fixture.state)

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
