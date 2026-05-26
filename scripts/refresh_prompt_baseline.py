"""Refresh ``tests/prompt_regression/llm_judge/baseline.json``.

Runs every metric on every ``expected_verdict == "pass"`` finding fixture,
takes the *minimum* observed score per metric, subtracts a small floor margin,
and writes the result back as the new threshold. The minimum (not the mean) is
deliberate: the threshold must be the floor every fixture clears, otherwise
the regression test isn't a regression test.

Run via::

    make refresh-prompt-baseline

Requires the ``eval`` extra and ``GEMINI_API_KEY``. Review the diff in
``baseline.json`` before committing — large downward moves indicate a real
regression that should be investigated, not papered over with a lower floor.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.prompt_regression.conftest import load_finding_fixtures  # noqa: E402
from tests.prompt_regression.llm_judge.gemini_judge import build_judge  # noqa: E402
from tests.prompt_regression.llm_judge.metrics import (  # noqa: E402
    build_metrics,
    finding_to_test_case,
)

BASELINE_PATH = REPO_ROOT / "tests/prompt_regression/llm_judge/baseline.json"
FLOOR_MARGIN = 0.05  # subtract from observed min so small jitter doesn't fail CI

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("refresh_prompt_baseline")


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY is not set; cannot refresh baseline.")
        return 1

    baseline = json.loads(BASELINE_PATH.read_text())
    # Start from current thresholds — the build_metrics call needs them, but
    # we override them below from observed scores.
    current_thresholds = baseline["thresholds"]

    judge = build_judge()
    metrics = build_metrics(judge, current_thresholds)

    fixtures = [f for f in load_finding_fixtures() if f.expected_verdict == "pass"]
    if not fixtures:
        logger.error("no `pass` finding fixtures found — nothing to baseline against.")
        return 1

    observed: dict[str, list[float]] = {name: [] for name in metrics}
    for fixture in fixtures:
        case = finding_to_test_case(fixture.finding)
        for name, metric in metrics.items():
            metric.measure(case)
            observed[name].append(metric.score)
            logger.info("  %s / %s -> %.3f", fixture.name, name, metric.score)

    new_thresholds: dict[str, float] = {}
    for name, scores in observed.items():
        floor = round(max(0.0, min(scores) - FLOOR_MARGIN), 3)
        logger.info("%s: min=%.3f n=%d -> threshold=%.3f", name, min(scores), len(scores), floor)
        new_thresholds[name] = floor

    baseline["thresholds"] = new_thresholds
    baseline.setdefault("_meta", {})["refreshed_at"] = datetime.now(UTC).isoformat()
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
    logger.info("wrote %s", BASELINE_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
