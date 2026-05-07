"""Shared JSON-parsing helper for self-eval and judge evaluators.

LLMs occasionally wrap JSON in markdown fences or add stray text. This helper
strips common wrappers, validates against EvaluationResult, and returns a safe
fallback verdict if parsing fails — so a malformed evaluator output downgrades
to PASS+WARN rather than crashing the run.
"""

from __future__ import annotations

import json
import logging
import re

from queryargus.models.evaluation import EvaluationResult, EvaluationVerdict

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def parse_evaluation_json(raw: str, *, evaluated_by: str) -> EvaluationResult:
    """Parse a raw LLM JSON response into ``EvaluationResult``.

    On any parse error returns a WARN verdict carrying the raw text in the
    reason — the run continues, the failure is visible in the audit trail.
    """
    text = raw.strip()
    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("evaluator returned non-JSON output: %s", exc)
        return _malformed(evaluated_by, f"non-json output: {exc}", raw)

    if not isinstance(data, dict):
        return _malformed(evaluated_by, "evaluator output was not a JSON object", raw)

    # The LLM may forget evaluated_by; fill it in deterministically.
    data.setdefault("evaluated_by", evaluated_by)

    try:
        return EvaluationResult.model_validate(data)
    except Exception as exc:  # noqa: BLE001 — surface any pydantic error as a warn
        logger.warning("evaluator output failed schema validation: %s", exc)
        return _malformed(evaluated_by, f"schema validation: {exc}", raw)


def _malformed(evaluated_by: str, reason: str, raw: str) -> EvaluationResult:
    snippet = raw[:200].replace("\n", " ")
    return EvaluationResult(
        verdict=EvaluationVerdict.WARN,
        score=0.5,
        reason=f"evaluator output unparseable ({reason}); raw[0:200]={snippet!r}",
        critique=None,
        evaluated_by=evaluated_by,
    )
