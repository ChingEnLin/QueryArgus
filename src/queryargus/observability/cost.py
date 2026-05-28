"""CostTracker — accumulates per-model token usage and writes RunCost to the report.

Pricing is in USD per 1,000,000 tokens, expressed as ``(input_rate, output_rate)``.
``DEFAULT_PRICING`` covers the models the QueryArgus suite uses today; callers
may override via the constructor when pricing changes.

Unknown models (including the empty-string "model unknown" case from clients
that don't report their model) contribute zero USD, but their token counts are
still recorded in ``RunCost.by_model`` so usage stays auditable.
"""

from __future__ import annotations

import logging
from typing import Literal
from uuid import UUID

from queryargus.llm.client import TokenUsage
from queryargus.models.report import AuditReport, ModelCost, RunCost

logger = logging.getLogger(__name__)


ModelPricing = dict[str, tuple[float, float]]


DEFAULT_PRICING: ModelPricing = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


class CostTracker:
    """Observer that turns LLM token usage into a ``RunCost`` on the report."""

    def __init__(
        self,
        pricing: ModelPricing | None = None,
        *,
        pricing_version: str = "2026-05",
    ) -> None:
        self._pricing: ModelPricing = (
            dict(pricing) if pricing is not None else dict(DEFAULT_PRICING)
        )
        self._pricing_version = pricing_version
        self._buckets: dict[str, TokenUsage] = {}
        self._warned_unknown: set[str] = set()

    def on_run_start(self, *, run_id: UUID, collection: str) -> None:
        self._buckets = {}
        self._warned_unknown = set()

    def on_llm_call(
        self,
        *,
        purpose: Literal["propose_action", "self_eval", "judge"],
        model: str,
        usage: TokenUsage,
        latency_ms: int,
    ) -> None:
        key = model
        prev = self._buckets.get(key, TokenUsage())
        self._buckets[key] = prev + usage
        if (key == "" or key not in self._pricing) and key not in self._warned_unknown:
            self._warned_unknown.add(key)
            logger.warning(
                "unknown model %r in CostTracker; tokens recorded with zero USD",
                key,
            )

    def on_run_complete(self, *, report: AuditReport) -> None:
        rows: list[ModelCost] = []
        total = 0.0
        for model, usage in self._buckets.items():
            in_rate, out_rate = self._pricing.get(model, (0.0, 0.0))
            usd = (
                usage.input_tokens * in_rate + usage.output_tokens * out_rate
            ) / 1_000_000.0
            total += usd
            rows.append(
                ModelCost(
                    model=model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    usd=usd,
                )
            )
        report.cost = RunCost(
            usd_total=total,
            by_model=rows,
            pricing_version=self._pricing_version,
        )
