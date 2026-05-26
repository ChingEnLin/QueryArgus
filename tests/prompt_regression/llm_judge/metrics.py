"""GEval metric definitions for the LLM-judged regression layer.

Two metrics in v1 — both finding-level, no report fixtures yet:

- ``severity_calibration``: does the severity match the magnitude and nature
  of the evidence (affected_pct, evidence_query)? Catches the "judge keeps
  downgrading severity" failure mode that motivated commit ``719b0de``.

- ``hypothesis_groundedness``: does the hypothesis follow from the evidence,
  or is it speculation? Catches the failure mode where the agent writes
  plausible-sounding causes that the evidence doesn't actually support.

Adding a metric is two steps: add a builder here, then add the metric name +
threshold to ``baseline.json``.
"""

from __future__ import annotations

from typing import Any

from queryargus.models.finding import Finding


def build_metrics(judge: Any, thresholds: dict[str, float]) -> dict[str, Any]:
    """Construct GEval metrics bound to a pinned judge LLM.

    Lazy-imports deepeval so this module is importable without the eval extra.
    Thresholds come from ``baseline.json`` — never hard-coded.
    """
    from deepeval.metrics import GEval  # noqa: PLC0415
    from deepeval.test_case import LLMTestCaseParams  # noqa: PLC0415

    severity_calibration = GEval(
        name="severity_calibration",
        evaluation_steps=[
            "Read the finding's severity, affected_pct, evidence_query, and description.",
            "CRITICAL requires affected_pct >= 0.01 AND unambiguous data corruption (impossible values, broken refs).",
            "HIGH is for clear quality issues with non-trivial impact.",
            "MEDIUM and LOW are for borderline or narrowly-scoped issues.",
            "Penalise CRITICAL on small affected_pct, or LOW on widespread corruption.",
            "Reward severities that match the magnitude AND nature of the evidence.",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=thresholds["severity_calibration"],
        model=judge,
        async_mode=False,
    )

    hypothesis_groundedness = GEval(
        name="hypothesis_groundedness",
        evaluation_steps=[
            "Read the finding's hypothesis, evidence_query, and sample_values.",
            "A grounded hypothesis names a specific mechanism that the evidence could plausibly support.",
            "Speculation that ignores or contradicts the evidence is ungrounded.",
            "Vague hypotheses like 'maybe optional' or 'probably user error' without specificity are ungrounded.",
            "Reward hypotheses that tie the suspected cause to the observed pattern.",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=thresholds["hypothesis_groundedness"],
        model=judge,
        async_mode=False,
    )

    return {
        "severity_calibration": severity_calibration,
        "hypothesis_groundedness": hypothesis_groundedness,
    }


def finding_to_test_case(finding: Finding) -> Any:
    """Convert a Finding into a deepeval LLMTestCase.

    INPUT carries the structural context (severity, affected_pct, evidence_query)
    so the judge sees what it needs to score. ACTUAL_OUTPUT carries the
    natural-language portions (description, hypothesis) that the metric judges.
    """
    from deepeval.test_case import LLMTestCase  # noqa: PLC0415

    input_text = (
        f"FIELD: {finding.field}\n"
        f"CATEGORY: {finding.category}\n"
        f"SEVERITY: {finding.severity.value}\n"
        f"AFFECTED_COUNT: {finding.affected_count}\n"
        f"AFFECTED_PCT: {finding.affected_pct:.6f}\n"
        f"EVIDENCE_QUERY: {finding.evidence_query}\n"
        f"SAMPLE_VALUES: {finding.sample_values!r}"
    )
    actual_output = (
        f"DESCRIPTION: {finding.description}\n"
        f"HYPOTHESIS: {finding.hypothesis}"
    )
    return LLMTestCase(input=input_text, actual_output=actual_output)
