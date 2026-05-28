from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import pytest

from queryargus.llm.client import TokenUsage
from queryargus.models.report import AuditReport
from queryargus.observability.cost import DEFAULT_PRICING, CostTracker


def _fresh_report() -> AuditReport:
    return AuditReport(
        collection="c",
        database="d",
        cosmos_account="a",
        duration_seconds=0.0,
        documents_sampled=0,
        collection_size=0,
    )


def test_cost_tracker_math_with_known_model() -> None:
    pricing = {"m1": (1.0, 2.0)}
    tracker = CostTracker(pricing=pricing, pricing_version="test")
    tracker.on_run_start(run_id=uuid4(), collection="c")
    tracker.on_llm_call(
        purpose="propose_action",
        model="m1",
        usage=TokenUsage(input_tokens=1_000_000, output_tokens=500_000),
        latency_ms=1,
    )
    tracker.on_llm_call(
        purpose="self_eval",
        model="m1",
        usage=TokenUsage(input_tokens=500_000, output_tokens=0),
        latency_ms=1,
    )
    report = _fresh_report()
    tracker.on_run_complete(report=report)
    assert report.cost is not None
    assert report.cost.pricing_version == "test"
    # 1.5M input * $1/M + 0.5M output * $2/M = 1.5 + 1.0 = 2.5
    assert report.cost.usd_total == pytest.approx(2.5)
    assert len(report.cost.by_model) == 1
    assert report.cost.by_model[0].model == "m1"
    assert report.cost.by_model[0].input_tokens == 1_500_000
    assert report.cost.by_model[0].output_tokens == 500_000


def test_cost_tracker_unknown_model_warns_once(caplog: Any) -> None:
    tracker = CostTracker(pricing={}, pricing_version="test")
    tracker.on_run_start(run_id=uuid4(), collection="c")
    with caplog.at_level(logging.WARNING, logger="queryargus.observability.cost"):
        tracker.on_llm_call(
            purpose="propose_action",
            model="mystery",
            usage=TokenUsage(input_tokens=100, output_tokens=100),
            latency_ms=1,
        )
        tracker.on_llm_call(
            purpose="propose_action",
            model="mystery",
            usage=TokenUsage(input_tokens=100, output_tokens=100),
            latency_ms=1,
        )
    warnings = [r for r in caplog.records if "mystery" in r.message]
    assert len(warnings) == 1

    report = _fresh_report()
    tracker.on_run_complete(report=report)
    assert report.cost is not None
    assert report.cost.usd_total == 0.0
    assert report.cost.by_model[0].model == "mystery"
    assert report.cost.by_model[0].usd == 0.0
    assert report.cost.by_model[0].input_tokens == 200


def test_cost_tracker_empty_model_string_treated_as_unknown() -> None:
    tracker = CostTracker(pricing={"m1": (1.0, 1.0)}, pricing_version="test")
    tracker.on_run_start(run_id=uuid4(), collection="c")
    tracker.on_llm_call(
        purpose="propose_action",
        model="",
        usage=TokenUsage(input_tokens=1_000_000, output_tokens=0),
        latency_ms=1,
    )
    report = _fresh_report()
    tracker.on_run_complete(report=report)
    assert report.cost is not None
    assert report.cost.usd_total == 0.0


def test_cost_tracker_resets_between_runs() -> None:
    tracker = CostTracker(pricing={"m1": (1.0, 1.0)}, pricing_version="test")
    tracker.on_run_start(run_id=uuid4(), collection="c")
    tracker.on_llm_call(
        purpose="propose_action",
        model="m1",
        usage=TokenUsage(input_tokens=1_000_000, output_tokens=0),
        latency_ms=1,
    )
    r1 = _fresh_report()
    tracker.on_run_complete(report=r1)

    tracker.on_run_start(run_id=uuid4(), collection="c")
    r2 = _fresh_report()
    tracker.on_run_complete(report=r2)
    assert r2.cost is not None
    assert r2.cost.usd_total == 0.0
    assert r2.cost.by_model == []


def test_default_pricing_has_expected_keys() -> None:
    assert "gemini-2.5-flash" in DEFAULT_PRICING
    assert "gemini-2.5-pro" in DEFAULT_PRICING
