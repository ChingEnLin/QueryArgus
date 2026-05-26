"""LLM-judged regression on well-formed finding fixtures.

What this catches:
- Calibration drift: a prompt edit that makes the judge systematically downgrade
  severity, or that lets ungrounded hypotheses sneak past.
- Fixture drift: someone weakens a "pass" fixture without realising it no longer
  represents a well-calibrated finding.

What this does **not** do:
- It does not score adversarial fixtures (those are the deterministic layer's
  job). Only ``expected_verdict == "pass"`` findings are passed to the judge.
- It does not gate PRs by default — run with ``-m prompt_regression_llm`` and
  ``GEMINI_API_KEY`` set. Default ``pytest`` skips the marker entirely.

Thresholds live in ``baseline.json`` and are refreshed only via
``make refresh-prompt-baseline``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.prompt_regression.conftest import FindingFixture, load_finding_fixtures

BASELINE_PATH = Path(__file__).parent / "baseline.json"


def _load_thresholds() -> dict[str, float]:
    data = json.loads(BASELINE_PATH.read_text())
    return data["thresholds"]


def _pass_only(fixtures: list[FindingFixture]) -> list[FindingFixture]:
    return [f for f in fixtures if f.expected_verdict == "pass"]


@pytest.fixture(scope="session")
def judge() -> Any:
    pytest.importorskip("deepeval", reason="install with: pip install -e '.[eval]'")
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set; LLM-judged regression layer skipped.")
    from .gemini_judge import build_judge  # noqa: PLC0415

    return build_judge()


@pytest.fixture(scope="session")
def thresholds() -> dict[str, float]:
    return _load_thresholds()


@pytest.fixture(scope="session")
def metrics(judge: Any, thresholds: dict[str, float]) -> dict[str, Any]:
    from .metrics import build_metrics  # noqa: PLC0415

    return build_metrics(judge, thresholds)


_FIXTURES = _pass_only(load_finding_fixtures())


@pytest.mark.prompt_regression_llm
@pytest.mark.parametrize("fixture", _FIXTURES, ids=[f.name for f in _FIXTURES])
def test_finding_clears_judge_thresholds(
    fixture: FindingFixture,
    metrics: dict[str, Any],
    thresholds: dict[str, float],
) -> None:
    from .metrics import finding_to_test_case  # noqa: PLC0415

    test_case = finding_to_test_case(fixture.finding)

    failures: list[str] = []
    for name, metric in metrics.items():
        metric.measure(test_case)
        floor = thresholds[name]
        if metric.score < floor:
            failures.append(
                f"{name}: score={metric.score:.2f} < threshold={floor:.2f} — {metric.reason}"
            )

    assert not failures, (
        f"fixture={fixture.name} failed LLM judge:\n  " + "\n  ".join(failures)
    )
